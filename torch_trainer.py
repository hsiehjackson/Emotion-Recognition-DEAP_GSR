import torch
import numpy as np
from logger import Logger
from torch_util import *
from torch_path import *
from sklearn.metrics import f1_score

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, File_Folder):
	logger = Logger(log_path+File_Folder)
	print(model)
	best_acc = 0.0
	for epoch in range(n_epochs):
		scheduler.step()		
		train_loss, train_acc, train_f1 = train_epoch(train_loader, model, loss_fn, optimizer, cuda)
		val_loss, val_acc, val_f1 = test_epoch(val_loader, model, loss_fn, cuda)

		message = '\rEpoch: {}/{} | Train loss: {:.4f} | Validation loss: {:.4f} | Train acc: {:.4f} | Train f1: {:.4f} | Validation acc: {:.4f} | Validation f1: {:.4f}'.format(
					epoch + 1, n_epochs, train_loss, val_loss, train_acc, train_f1, val_acc, val_f1)
		print(message,end='')

		if (val_acc > best_acc):
			best_acc = val_acc
			torch.save(model.state_dict(), ckpt_path+File_Folder+'/'+str(val_acc)+'.pkt')
			print ('\nSave Improved Model(val_acc = %.6f)...' % (val_acc))
		else:
			print('')

		info = { 'train_loss': train_loss, 'train_acc':train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'train_f1': train_f1, 'val_f1': val_f1}
		for tag, value in info.items():
			logger.scalar_summary(tag, value, epoch+1)

def train_epoch(train_loader, model, loss_fn, optimizer, cuda):

	model.train()
	train_loss = 0
	train_acc = 0
	train_f1 = 0
	start = time.time()

	for batch_idx, (b_x, b_y) in enumerate(train_loader):
		b_x = b_x.view(len(b_x),len(b_x[0]),-1).cuda()
		b_y = b_y.cuda()
		optimizer.zero_grad()
		output = model(b_x)
		batch_loss = loss_fn(output, b_y)
		batch_loss.backward()
		optimizer.step()

		train_loss += batch_loss.item()
		output_label = torch.argmax(output,1).cpu()
		Acc = np.sum((output_label == b_y.cpu()).numpy())
		f1 = f1_score(b_y.cpu(),output_label,average='binary')
		train_acc += Acc
		train_f1 += f1

		if batch_idx > 0:
			message = '\r[{}/{} ({:.0f}%)] | Time: {} | Loss: {:.6f} | Acc: {:.6f} | f1: {:.6f}'.format(
						batch_idx * len(b_x[0]), 
						len(train_loader.dataset),
						100. * batch_idx / len(train_loader), 
						timeSince(start, batch_idx/len(train_loader)),
						batch_loss.item(),
						Acc/len(b_x),
						f1)

			print(message,end='')

	train_loss /= (len(train_loader))
	train_f1 /= (len(train_loader))
	train_acc /= (len(train_loader.dataset))
	return train_loss, train_acc, train_f1

def test_epoch(val_loader, model, loss_fn, cuda):
	with torch.no_grad():
		model.eval()
		val_loss = 0
		val_acc = 0
		val_f1 = 0
		for batch_idx, (b_x, b_y) in enumerate(val_loader):
			b_x = b_x.view(len(b_x),len(b_x[0]),-1).cuda()
			b_y = b_y.cuda()
			output = model(b_x)

			batch_loss = loss_fn(output, b_y)
			val_loss += batch_loss.item()
			output_label = torch.argmax(output,1).cpu()
			Acc = np.sum((output_label== b_y.cpu()).numpy())
			f1 = f1_score(b_y.cpu(),output_label,average='binary')
			val_acc += Acc
			val_f1 += f1

	val_loss /= len(val_loader)
	val_acc /= len(val_loader.dataset)
	val_f1 /= len(val_loader)
	return val_loss, val_acc, val_f1