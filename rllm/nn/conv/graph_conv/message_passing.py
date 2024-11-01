import torch
from torch import Tensor
from torch_scatter import scatter
from torch_sparse import SparseTensor
from typing import Optional, List, Tuple, Union, Any, Callable

class MessagePassing(torch.nn.Module):
    def __init__(self, aggr: str = "add", flow: str = "source_to_target"):
        super(MessagePassing, self).__init__()
        self.aggr = aggr
        self.flow = flow
        self.__user_args__ = self.inspector.keys(['message', 'aggregate', 'update'])


def propagate(self, edge_index, size: Optional[Tuple[int, int]] = None, **kwargs):
    size = self.__check_input__(edge_index, size)
    coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
     
    msg_kwargs = self.inspector.distribute('message', coll_dict)
    out = self.message(**msg_kwargs)
     
    aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
    out = self.aggregate(out, **aggr_kwargs)

    update_kwargs = self.inspector.distribute('update', coll_dict)
    out = self.update(out, **update_kwargs)
 
    return out


def message(self, x_j: Tensor) -> Tensor:   
    return x_j


def aggregate(self, inputs: Tensor, index: Tensor, dim_size: Optional[int] = None) -> Tensor:
    return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)



def update(self, aggr_out: Tensor) -> Tensor:
    return aggr_out


def __check_input__(self, edge_index, size):
    return size

