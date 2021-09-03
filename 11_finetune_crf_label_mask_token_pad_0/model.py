from torch import nn
from transformers import ElectraModel, ElectraPreTrainedModel
from CRF import CRF
from transformers.modeling_outputs import TokenClassifierOutput

class KoelectraCRF(ElectraPreTrainedModel):
    def __init__(self, config):
        super(KoelectraCRF, self).__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_wise_ff = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, label_mask=None):
        outputs =self.electra(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.position_wise_ff(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=label_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores

