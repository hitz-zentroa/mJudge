#from transformers.loss.loss_utils import ForCausalLMLoss
import torch
from colorama import Fore, Style
import logging
import sys
from transformers.loss.loss_utils import ForCausalLMLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_subsequence_last(seq, subseq):
    """FINISHED for labels since shape (bs, seq_len)"""

    real_idx = [i for i, token in enumerate(seq) if token.item() != -100] # idx of non 0 elements
    valid_tokens = [token.item() for token in seq if token.item() != -100]

    score_idxs=[]
    for i in range(len(valid_tokens) - len(subseq) + 1):
        if valid_tokens[i:i+len(subseq)] == subseq:
            score_idxs.append(real_idx[i+len(subseq)]) # append absolute idx
    return score_idxs


def get_tokens(sequences, result_tokens):
    """ for labels since shape bs, seq_len"""
    indices_to_scores = {
        16: 1.0,
        17: 2.0,
        18: 3.0,
        19: 4.0,
        20: 5.0,
    }
    masked_seq = []
    score_indexes=[]
    scores=[]
    for seq in sequences:
        # seq = torch. tensor of seq len
        sc_idx = find_subsequence_last(seq,result_tokens)
        score_indexes.append(sc_idx)
        scores.append([indices_to_scores[seq[id].item()] for id in sc_idx]) #take only number tokens. len o this should be 1 acually

        for idx in sc_idx:
            seq[idx]=-100 #Mask the score
        masked_seq.append(seq)
    
    return torch.stack(masked_seq),scores, score_indexes



def compute_raft_loss(logits, labels ,tokenizer, num_items_in_batch=None):
    
    result_tokens=tokenizer(" [RESULT] ", add_special_tokens=False)["input_ids"] #[510, 14430, 60,220] # Hamen pasa izetez el token del entrnamiento hau en mono no va.
    masked_label_logs,detokenized_score_labels,score_indexes = get_tokens(labels,result_tokens)

    print("Masked label shape",masked_label_logs.shape)
    print("logs shape",logits.shape)

    print("Scores",detokenized_score_labels)
    print("IDX",score_indexes)

    # FEEDBACK

    # Step 3: Compute the LM loss
    # TODO: The num_items_in_batch is wrong since we mask out the score label. It should be substracted by something multiply by the world size?

    #lm_loss=torch.nn.CrossEntropyLoss()(logits,masked_label_logs)

    lm_loss = ForCausalLMLoss(
        logits = logits, 
        labels = masked_label_logs,
        vocab_size = logits.size(-1),
        #num_items_in_batch = num_items_in_batch - num_seq, # TODO: Modify this
    )
    print("Crossentropy loss: ",lm_loss)
    #lm_loss=torch.nn.CrossEntropyLoss()(torch.softmax(logits, dim=-1),masked_label_logs)
    #print("otherway loss: ",lm_loss)

    #SCORE

    #detokenized_score_labels=torch.tensor([float(tokenizer.decode(x, skip_special_tokens=True)) for x in label_scores])
    #print("Labels",detokenized_score_labels)

    # Step 4: Compute the score loss
    # Seq len 5
    # Token pos: 0 1 2 3 4
    # Input    : A B C D E
    # Is score : x x x v x
    # Predict  : B C D E - 
    # We take -1 due to the shift between input and output
    
    score_logits = []
    for seq, ids in zip(logits,score_indexes):
        score_logits.append(seq[ids, :])
    #score_logits = torch.stack(score_logits) no tienen el mismo shape posiblemente

    # probs=[]
    # for element,id in zip(logits,score_indexes):
    #     print(element[id-1].shape)
    #     probs.append(element[id-1])
    #     #s=float(tokenizer.decode(element[id-1].argmax(), skip_special_tokens=True))
    #     #predicted_scores.append(s)
    # probs=torch.tensor(probs)
    # print("probs shape :",probs.shape)

    score_indices = [16, 17, 18, 19, 20]
    posible_scores = [1.0, 2.0, 3.0, 4.0, 5.0]

    score_probs = detokenized_score_labels.contiguous() # 

    
    probs = [torch.softmax(log, dim=-1) for log in score_logits] # Shape: (batch_size, vocab_size). Desberdiña nire acsun the pach se virtualmente es un batch más grance
    score_probs = [prob[..., score_indices].contiguous() for prob in probs]# 
    # Compute the weighted sum of the score
    weighted_scores = [torch.sum(
        prob * torch.tensor(posible_scores),
        dim = -1,
        keepdim = False,
    ) for prob in score_probs]
    print("Weighted scores",weighted_scores)

    # logger.info(f"{Fore.GREEN}score_label_token_ids:{Style.RESET_ALL} {score_label_token_ids}")
    # logger.info(f"{Fore.GREEN}score_labels:{Style.RESET_ALL} {score_labels}")
    # logger.info(f"{Fore.GREEN}score_grid_probs:{Style.RESET_ALL} {score_grid_probs}")
    # logger.info(f"{Fore.GREEN}weighted_scores:{Style.RESET_ALL} {weighted_scores}")

    # Compute the MSE loss
    score_loss = torch.nn.functional.mse_loss(
        input = weighted_scores, 
        target = detokenized_score_labels,
        reduction = 'sum' if num_items_in_batch is None else 'mean',
    )
    print("Score loss: ",score_loss)

    if num_items_in_batch is not None:
        print("if barru")
        score_loss = score_loss / labels.size(0) # TODO: This should be the number of sequences in the whole batch (I am not sure whether we should consider the world size)
    # TODO: Find a way to log the loss
    loss = lm_loss +  1.0 * score_loss
    print(f"LM loss: {Fore.BLUE}{lm_loss.item():.4f}{Style.RESET_ALL}, Score loss: {Fore.BLUE}{score_loss.item():.4f}{Style.RESET_ALL}")
    return loss