import logging
import tensorflow as tf
from .compressor import Pruner

__all__ = ['LevelPruner', 'AGP_Pruner', 'FilterPruner', 'SlimPruner']

_logger = logging.getLogger(__name__)


class LevelPruner(Pruner):
    """Prune to an exact pruning level specification
    """

    def __init__(self, config_list):
        """
        config_list: supported keys:
            - sparsity
        """
        super().__init__(config_list)
        self.now_epoch = tf.Variable(0)

    def calc_mask(self, weight, config, op_name, **kwargs):
        threshold = tf.contrib.distributions.percentile(tf.abs(weight), config['sparsity'] * 100)
        mask = tf.cast(tf.math.greater(tf.abs(weight), threshold), weight.dtype)
        return mask


class AGP_Pruner(Pruner):
    """An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.

    Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
    efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
    Learning of Phones and other Consumer Devices,
    https://arxiv.org/pdf/1710.01878.pdf
    """

    def __init__(self, config_list):
        """
        config_list: supported keys:
            - initial_sparsity
            - final_sparsity: you should make sure initial_sparsity <= final_sparsity
            - start_epoch: start epoch numer begin update mask
            - end_epoch: end epoch number stop update mask
            - frequency: if you want update every 2 epoch, you can set it 2
        """
        super().__init__(config_list)
        self.mask_list = {}
        self.if_init_list = {}
        self.now_epoch = tf.Variable(0)
        self.assign_handler = []

    def calc_mask(self, weight, config, op_name, **kwargs):
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        if self.now_epoch >= start_epoch and self.if_init_list.get(op_name, True) and (
                    self.now_epoch - start_epoch) % freq == 0:
            target_sparsity = self.compute_target_sparsity(config)
            threshold = tf.contrib.distributions.percentile(weight, target_sparsity * 100)
            # stop gradient in case gradient change the mask
            mask = tf.stop_gradient(tf.cast(tf.math.greater(weight, threshold), weight.dtype))
            self.assign_handler.append(tf.assign(weight, weight * mask))
            self.mask_list.update({op_name: tf.constant(mask)})
            self.if_init_list.update({op_name: False})
        else:
            mask = self.mask_list[op_name]
        return mask

    def compute_target_sparsity(self, config):
        end_epoch = config.get('end_epoch', 1)
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        final_sparsity = config.get('final_sparsity', 0)
        initial_sparsity = config.get('initial_sparsity', 0)

        if end_epoch <= start_epoch or initial_sparsity >= final_sparsity:
            _logger.warning('your end epoch <= start epoch or initial_sparsity >= final_sparsity')
            return final_sparsity

        now_epoch = tf.minimum(self.now_epoch, tf.constant(end_epoch))
        span = int(((end_epoch - start_epoch - 1) // freq) * freq)
        assert span > 0
        base = tf.cast(now_epoch - start_epoch, tf.float32) / span
        target_sparsity = (final_sparsity +
                           (initial_sparsity - final_sparsity) *
                           (tf.pow(1.0 - base, 3)))
        return target_sparsity

    def update_epoch(self, epoch, sess):
        sess.run(self.assign_handler)
        sess.run(tf.assign(self.now_epoch, int(epoch)))
        for k in self.if_init_list.keys():
            self.if_init_list[k] = True


class FilterPruner(Pruner):
    """A structured pruning algorithm that prunes the filters of smallest magnitude
    weights sum in the convolution layers to achieve a preset level of network sparsity.

    Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf,
    "PRUNING FILTERS FOR EFFICIENT CONVNETS", 2017 ICLR
    https://arxiv.org/abs/1608.08710
    """

    def __init__(self, config_list):
        """
        config_list: supported keys:
            - sparsity
        """
        super().__init__(config_list)
        self.mask_list = {}
        self.if_init_list = {}

    def calc_mask(self, weight, config, op_name, **kwargs):
        if self.if_init_list.get(op_name, True):
            threshold = tf.contrib.distributions.percentile(tf.reduce_mean(tf.abs(weight), [1, 2, 3]),
                                                            config['sparsity'] * 100)
            mask = tf.cast(tf.math.greater(tf.reduce_mean(tf.abs(weight), [1, 2, 3]), threshold), weight.dtype)
            self.mask_list.update({op_name: mask})
            self.if_init_list.update({op_name: False})
        else:
            mask = self.mask_list[op_name]
        return mask


class SlimPruner(Pruner):
    """A structured pruning algorithm that prunes channels by pruning the weights of BN layers

    Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan and Changshui Zhang
    "Learning Efficient Convolutional Networks through Network Slimming", 2017 ICCV
    https://arxiv.org/pdf/1708.06519.pdf
    """

    def __init__(self, config_list):
        """
        config_list: supported keys:
            - sparsity
        """
        super().__init__(config_list)
        self.mask_list = {}
        self.if_init_list = {}

    def bind_model(self, model):
        weight_list = []
        config = self._config_list[0]
        op_types = config.get('op_types')
        op_names = config.get('op_names')
        if op_types is not None:
            assert op_types is 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
            for name, m in model.named_modules():
                if type(m).__name__ is 'BatchNorm2d':
                    weight_list.append(m.weight.data.clone())
        else:
            for name, m in model.named_modules():
                if name in op_names:
                    assert type(
                        m).__name__ is 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
                    weight_list.append(m.weight.data.clone())
        all_bn_weights = torch.cat(weight_list)
        k = int(all_bn_weights.shape[0] * config['sparsity'])
        self.global_threshold = torch.topk(all_bn_weights.view(-1), k, largest=False).values.max()

    def calc_mask(self, weight, config, op_name, op_type, **kwargs):
        assert op_type is 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
        if self.if_init_list.get(op_name, True):
            w_abs = weight.abs()
            mask = torch.gt(w_abs, self.global_threshold).type_as(weight)
            self.mask_list.update({op_name: mask})
            self.if_init_list.update({op_name: False})
        else:
            mask = self.mask_list[op_name]
        return mask
