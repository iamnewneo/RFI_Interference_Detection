# import torch
# import numpy as np
# from tqdm import tqdm


# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1)
#     return np.sum(outputs == labels)


# def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
#     model = model.train()
#     losses = []
#     accuracy_sum = 0
#     for batch in tqdm(data_loader):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         token_type_ids = batch["token_type_ids"].to(device)
#         c_len = batch["c_len"].to(device)
#         q_len = batch["q_len"].to(device)
#         a_len = batch["a_len"].to(device)
#         targets = batch["target"].type(dtype=torch.long).to(device)

#         outputs = model(
#             input_ids=input_ids,
#             token_type_ids=token_type_ids,
#             attention_mask=attention_mask,
#             doc_len=c_len,
#             ques_len=q_len,
#             option_len=a_len,
#             labels=targets,
#         )

#         loss = loss_fn(outputs, targets)
#         outputs = outputs.detach().cpu().numpy()
#         targets = targets.to("cpu").numpy()
#         accuracy_sum += accuracy(outputs, targets)
#         losses.append(loss.item())
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()
#     return accuracy_sum / len(data_loader), np.mean(losses)


# def eval_model(model, data_loader, loss_fn, device):
#     model = model.eval()
#     losses = []
#     accuracy_sum = 0
#     with torch.no_grad():
#         for batch in tqdm(data_loader):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             token_type_ids = batch["token_type_ids"].to(device)
#             c_len = batch["c_len"].to(device)
#             q_len = batch["q_len"].to(device)
#             a_len = batch["a_len"].to(device)
#             targets = batch["target"].type(dtype=torch.long).to(device)

#             outputs = model(
#                 input_ids=input_ids,
#                 token_type_ids=token_type_ids,
#                 attention_mask=attention_mask,
#                 doc_len=c_len,
#                 ques_len=q_len,
#                 option_len=a_len,
#                 labels=targets,
#             )
#             loss = loss_fn(outputs, targets)
#             outputs = outputs.to("cpu").numpy()
#             targets = targets.to("cpu").numpy()
#             accuracy_sum += accuracy(outputs, targets)
#             losses.append(loss.item())
#     return accuracy_sum / len(data_loader), np.mean(losses)