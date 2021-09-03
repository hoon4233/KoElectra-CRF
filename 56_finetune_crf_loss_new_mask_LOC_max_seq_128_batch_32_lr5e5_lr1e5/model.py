from torch import nn
from transformers import ElectraModel, ElectraPreTrainedModel
# from CRF import CRF
from CRF2 import CRF
from transformers.modeling_outputs import TokenClassifierOutput

class KoelectraCRF(ElectraPreTrainedModel):
    def __init__(self, config):
        super(KoelectraCRF, self).__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_wise_ff = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, label_mask=None, crf_mask=None):
        outputs =self.electra(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.position_wise_ff(sequence_output)
        # outputs = (logits,)
        outputs = {}
        outputs['logits'] = logits

        mask = crf_mask
        batch_size = logits.shape[0]

        if labels is not None:
            loss = 0
            for seq_logits, seq_labels, seq_mask in zip(logits, labels, mask):
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(emissions = seq_logits, tags=seq_labels, reduction='token_mean')
            loss /= batch_size
            outputs['loss'] = loss

        # else :
        #     output_tags = []
        #     for seq_logits, seq_mask in zip(logits, mask):
        #         seq_logits = seq_logits[seq_mask].unsqueeze(0)
        #         tags = self.crf.decode(seq_logits)
        #         output_tags.append(tags[0])
        #     outputs['y_pred'] = output_tags
        output_tags = []
        for seq_logits, seq_mask in zip(logits, mask):
            seq_logits = seq_logits[seq_mask].unsqueeze(0)
            tags = self.crf.decode(seq_logits)
            output_tags.append(tags[0])
        outputs['y_pred'] = output_tags

        return outputs

