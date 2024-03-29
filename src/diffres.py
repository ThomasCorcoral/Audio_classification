import torch, os
import matplotlib.pyplot as plt

from src.core import Base
from src.conv import DilatedConv

from src.pooling import Pooling_layer

EPS = 1e-12

class DiffRes(Base):

    def __init__(self, in_t_dim, in_f_dim, dimension_reduction_rate, learn_pos_emb=False):
        super().__init__(in_t_dim, in_f_dim, dimension_reduction_rate, learn_pos_emb)
        self.feature_channels = 3
        self.current = 0

        self.model = DilatedConv(
            in_channels=self.input_f_dim,
            dilation_rate=1,
            input_size=self.input_seq_length,
            kernel_size=5,
            stride=1,
        )

    def forward(self, x):
        ret = {}
        score = torch.sigmoid(self.model(x.permute(0, 2, 1)).permute(0, 2, 1))
        score, _ = self.score_norm(score, self.output_seq_length)
        mean_feature, max_pool_feature, mean_pos_enc = self.frame_warping(
            x.exp(), score, total_length=self.output_seq_length
        )

        mean_feature = torch.log(mean_feature + EPS)
        max_pool_feature = torch.log(max_pool_feature + EPS)

        ret["x"] = x
        ret["score"] = score
        ret["resolution_enc"] = mean_pos_enc
        ret["avgpool"] = mean_feature
        ret["maxpool"] = max_pool_feature
        ret["feature"] = torch.cat(
            [
                mean_feature.unsqueeze(1),
                max_pool_feature.unsqueeze(1),
                mean_pos_enc.unsqueeze(1),
            ],
            dim=1,
        )
        ret["guide_loss"], ret["activeness"] = self.guide_loss(
            x, importance_score=score
        )

        #del score, mean_feature, max_pool_feature, mean_pos_enc
        #gc.collect()

        return ret

    def frame_warping(self, feature, score, total_length):
        weight = self.calculate_weight(score, feature, total_length=total_length)

        mean_feature = torch.matmul(weight.permute(0, 2, 1), feature)
        max_pool_feature = self.calculate_scatter_maxpool_odd_even_lines(
            weight, feature, out_len=self.output_seq_length
        )
        mean_pos_enc = torch.matmul(weight.permute(0, 2, 1), self.pos_emb)

        return mean_feature, max_pool_feature, mean_pos_enc

    def visualize(self, ret, savepath="./images_gen/"):
        x, y, emb, score = ret["x"], ret["feature"], ret["resolution_enc"], ret["score"]
        y = y[:, 0, :, :]
        nb_img_to_visualize = 10
        for i in range(nb_img_to_visualize):
            if(i >= x.size(0)):
                break
            plt.figure(figsize=(8, 16))
            plt.subplot(411)
            plt.title("Importance score")
            plt.plot(score[i, :, 0].detach().cpu().numpy())
            plt.ylim([0,1])
            plt.subplot(412)
            plt.title("Original mel spectrogram")
            plt.imshow(
                x[i, ...].detach().cpu().numpy().T, aspect="auto", interpolation="none"
            )
            plt.subplot(413)
            plt.title(
                "DiffRes mel spectrogram"
            )
            plt.imshow(
                y[i, ...].detach().cpu().numpy().T, aspect="auto", interpolation="none"
            )
            plt.subplot(414)
            plt.title("Resolution encoding")
            plt.imshow(
                emb[i, ...].detach().cpu().numpy().T,
                aspect="auto",
                interpolation="none",
            )
            path = savepath + str(self.current)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()
        self.current += 1