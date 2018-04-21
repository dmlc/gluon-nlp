from mxnet.gluon.loss import SoftmaxCELoss


class SoftmaxCEMaskedLoss(SoftmaxCELoss):
    """Wrapper of the SoftmaxCELoss that supports valid_length as the input

    """
    def hybrid_forward(self, F, pred, label, valid_length):
        """

        Parameters
        ----------
        F
        pred : Symbol or NDArray
            Shape (batch_size, length, V)
        label : Symbol or NDArray
            Shape (batch_size, length)
        valid_length : Symbol or NDArray
            Shape (batch_size, )
        Returns
        -------
        loss : Symbol or NDArray
            Shape (batch_size,)
        """
        sample_weight = F.expand_dims(F.ones_like(label), axis=-1)
        sample_weight = F.SequenceMask(sample_weight,
                                       sequence_length=valid_length,
                                       use_sequence_length=True,
                                       axis=1)
        return super(SoftmaxCEMaskedLoss, self).hybrid_forward(F, pred, label, sample_weight)
