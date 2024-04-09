import torch
from lightly.models.modules.heads import ProjectionHead


class TFCProjectionHead(ProjectionHead):
    def __init__(
        self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128
    ):
        super().__init__(
            [
                (
                    input_dim,
                    hidden_dim,
                    torch.nn.BatchNorm1d(256),
                    torch.nn.ReLU(),
                ),
                (hidden_dim, output_dim, None, None),
            ]
        )


class TFCPredictionHead(ProjectionHead):
    def __init__(
        self,
        input_dim: int = 2 * 128,
        hidden_dim: int = 64,
        output_dim: int = 2,
    ):
        super().__init__(
            [
                (
                    input_dim,
                    hidden_dim,
                    None,
                    torch.nn.Sigmoid(),
                ),
                (hidden_dim, output_dim, None, None),
            ]
        )


#ADAPTAÇÕES PARA CPCPredictionHead DO ARTIGO DO CPC PARA HAR

class TNCPredictionHead(ProjectionHead):
    def __init__(
        self,
        input_dim: int = 60,
        hidden_dim1: int = 150,
        hidden_dim2: int = 128,
        output_dim: int = 7,
        dropout_prob: float = 0.2,
    ):
        super().__init__(
            [
                (
                    input_dim,
                    hidden_dim1,
                    None,
                    torch.nn.BatchNorm1d(hidden_dim1),
                    torch.nn.Sequential(
                        torch.nn.ReLU(inplace=True), torch.nn.Dropout(p=dropout_prob),
                    ),
                ),
                (
                    hidden_dim1,
                    hidden_dim2,
                    None,
                    torch.nn.BatchNorm1d(hidden_dim2),
                    torch.nn.Sequential(
                        torch.nn.ReLU(inplace=True), torch.nn.Dropout(p=dropout_prob)
                    ),
                ),
                (
                    hidden_dim2,
                    output_dim,
                    None,
                    torch.nn.Softmax(dim=1),
                ),
            ]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        print('TNCPredictionHead x:::::::::', x.size())
        return x

class CPCPredictionHead(TNCPredictionHead):
    pass

class ConvAEPredictionHead(TNCPredictionHead):
    pass
