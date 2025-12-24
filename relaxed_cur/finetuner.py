from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class RelaxedCURFineTuner:
    def __init__(
        self,
        learning_rate: float = 1e-5,
        num_steps: int = 300,
        distill_weight: float = 0.5,
        warmup_steps: int = 30,
    ):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.distill_weight = distill_weight
        self.warmup_steps = warmup_steps
    
    def finetune(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        calib_data: List[Dict],
        device: str,
    ) -> Dict:
        student_model.train()
        teacher_model.eval()
        
        params_to_train = []
        for name, param in student_model.named_parameters():
            if any(x in name for x in ['C_eff', 'R_eff', '.U', 'gate_weight']):
                params_to_train.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        print(f"    Training {len(params_to_train)} parameter groups")
        
        optimizer = torch.optim.AdamW(params_to_train, lr=self.learning_rate)
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        log = {'lm_loss': [], 'distill_loss': [], 'total_loss': []}
        
        data_iter = iter(calib_data * ((self.num_steps // len(calib_data)) + 1))
        
        pbar = tqdm(range(self.num_steps), desc="    Fine-tuning")
        
        for step in pbar:
            batch = next(data_iter)
            input_ids = batch['input_ids'].to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids, use_cache=False)
                teacher_logits = teacher_outputs.logits
            
            student_outputs = student_model(input_ids, labels=input_ids, use_cache=False)
            lm_loss = student_outputs.loss
            student_logits = student_outputs.logits
            
            temperature = 2.0
            student_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            distill_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
            distill_loss = distill_loss * (temperature ** 2)
            
            total_loss = (1 - self.distill_weight) * lm_loss + self.distill_weight * distill_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, 1.0)
            optimizer.step()
            scheduler.step()
            
            log['lm_loss'].append(lm_loss.item())
            log['distill_loss'].append(distill_loss.item())
            log['total_loss'].append(total_loss.item())
            
            pbar.set_postfix({
                'lm': f"{lm_loss.item():.3f}",
                'kd': f"{distill_loss.item():.3f}",
            })
        
        student_model.eval()
        return log
