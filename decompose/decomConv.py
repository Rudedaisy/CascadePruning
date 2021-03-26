###################################################################
#   Defined Decomposed Convolution layer
#
#   UNFINISHED RESEARCH CODE
#   DO NOT DISTRIBUTE
#
#   Author: Shiyu Li
#   Date:   12/02/2019
##################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.cluster import KMeans

from decompose.fb import calculate_FB_bases
#from decompose.pyinn import conv2d_depthwise
#from decompose.rawCUDA import *
from collections import namedtuple
#import cupy

from sklearn.decomposition import PCA

#import sys
#sys.path.insert(1, '/root/Filter_Decompose/python/decompose')
#from extract import exportIFM
#import pandas as pd

Stream = namedtuple('Stream', ['ptr'])

#import common
#COUNT = 1

class DecomposedConv2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
				 dilation=1, device=None, bias=None, init='rand', num_basis=2,):
		super(DecomposedConv2D, self).__init__()
		if isinstance(kernel_size, int):    #If input only one kernel size
			kernel_size = [kernel_size, kernel_size]
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.num_basis = num_basis
		self.device = device

		self.bias = bias

		if init == 'rand':
			self.basis = nn.Parameter(torch.randn((num_basis, kernel_size[0] * kernel_size[1])), requires_grad=False)
		elif init == 'FB':
			if kernel_size[0] % 2 == 0:
				raise Exception('Kernel size for FB initialization only supports odd number for now.')
			base_np, _, _ = calculate_FB_bases(int((kernel_size[0] - 1) / 2))
			if num_basis > base_np.shape[1]:
				raise Exception(
					'The maximum number of bases for kernel size = %d is %d' % (kernel_size, base_np.shape[1]))
			elif num_basis == -1:
				num_basis = base_np.shape[1]
			else:
				base_np = base_np[:, :num_basis]
			base_np = base_np.reshape(num_basis, kernel_size[0] * kernel_size[1]).astype(np.float32)
			self.basis = nn.Parameter(torch.tensor(base_np), requires_grad=False)
		else:
			raise Exception("Unsupported initialization method!")
		self.coefs = nn.Parameter(torch.randn((out_channels * in_channels, num_basis)), requires_grad=True)
		#self.scales = nn.Parameter(torch.randn(self.out_channels * self.num_basis), requires_grad=True)


	def init_decompose_with_pca(self, basis, coefs):
		self.basis = nn.Parameter(torch.tensor(basis.reshape(self.num_basis, self.kernel_size[0] * self.kernel_size[1])), requires_grad=False)
		self.coefs = nn.Parameter(torch.tensor(coefs.reshape(self.out_channels * self.in_channels, self.num_basis)), requires_grad=True)

	def to_sparse(self):
		'''Convert Coefficient Matrix Into Sparse Format'''
		if isinstance(self.coefs.data, torch.sparse.FloatTensor):
			print("Already A Sparse Coefficient")
			return
		temp = self.coefs.data.view((self.out_channels, -1)).detach().cpu() #For Forward MM operation
		self.values = temp[temp!=0]
		self.indices = torch.nonzero(temp).T.numpy().tolist()

		self.row_idx = []
		self.col_idx = []
		self.acc_ptr = [0]
		for i in range(len(self.indices)):
			self.col_idx.append(np.unique(self.indices[i]))
			self.row_idx.append(np.ones_like(self.col_idx[-1]) * i)
			self.acc_ptr.append(len(self.col_idx[-1]))		#Indicate the position of each filter map
		#Flatten
		self.row_idx = torch.tensor(np.concatenate(self.row_idx).ravel()).cuda()
		self.col_idx = torch.tensor(np.concatenate(self.col_idx).ravel()).cuda()
		self.acc_ptr = torch.tensor(self.acc_ptr)
		#self.coefs = nn.Parameter(torch.sparse.FloatTensor(indices,value,temp.size()))

	def to_dense(self):
		if isinstance(self.coefs.data, torch.Tensor):
			print("Already A Dense Coefficient")
			return
		self.coefs = nn.Parameter(self.coefs.data.to_dense())
		self.coefs = self.coefs.view(-1, self.num_basis)

	def compile_CUDA_kernel(self, cuda_threads=1024):


		#sIFM = cupy.zeros((self.out_h * self.out_w *len(self.values), ), dtype=cupy.float32)

		if not hasattr(self, 'scaling') or not hasattr(self, 'accmulate'):
			n_threads = self.out_channels		#Thread Number is the output channels
			self.scaling = load_kernel("scaleBIFM", scaleCUDA,
									   coef_len= int(self.values.shape[0]),
									   H = int(self.out_h),
									   W = int(self.out_w),
									   HW = int(self.out_w * self.out_h),
									   DHW = int(self.num_basis * self.out_w * self.out_h))

			self.accumulate = load_kernel("accSIFM", accumulateCUDA,
									   coef_len= int(self.values.shape[0]),
									   C = int(self.out_channels),
									   W = int(self.out_w),
									   H = int(self.out_h),
									   HW = int(self.out_w * self.out_h))

	def oc_factorize(self):
		coefs = self.coefs.detach().cpu().numpy().reshape((self.out_channels,self.in_channels, self.num_basis))
		ic_vecs = []
		k_vecs = []
		for oc in range(coefs.shape[0]):
			pca = PCA(n_components=1)
			local_matrix = coefs[oc, :, :]
			ic_vec = pca.fit_transform(local_matrix).reshape((-1, 1))
			k_vec = pca.components_.reshape((1, -1))
			ic_vecs.append(ic_vec)
			k_vecs.append(k_vec)
		ic_vecs = np.array(ic_vecs)
		k_vecs = np.array(k_vecs)
		self.ic_vecs = nn.Parameter(torch.tensor(ic_vecs).cuda(), requires_grad=True)
		self.k_vecs = nn.Parameter(torch.tensor(k_vecs).cuda(), requires_grad=True)
		self.forward = self.forward_oc

	# def eval(self):
    #             self.forward = self.forward
		#self.to_sparse()
		#if not hasattr(self, 'out_h'):
		#	print("Please run model once to obtain the essential dimension data.")
		#	self.forward = self.forward_pre
		#else:
			#print("Compiling CUDA kernels...")
			#self.compile_CUDA_kernel()
		#	self.forward = self.forward_test


	# def train(self, mode=True):
	# 	if mode:
	# 		self.forward = self.forward_train
	# 	else:
	# 		self.eval()
	def fake_quant_coef(self, coefs=True, Basis=False, bits_basis = 8, bits_coef = 8, int_bits_basis = 1, int_bits_coef=1):
		#Fixed Point
		with torch.no_grad():
			if coefs:
				mul_coeff_coef = 2 ** (bits_coef - 1 - int_bits_coef )
				div_coeff_coef = 2 ** (int_bits_coef - bits_coef + 1)
				max_coeff_coef = 2 ** (bits_coef - 1)
				self.coefs = self.coefs.mul_(mul_coeff_coef).floor_().clamp_(-max_coeff_coef - 1, max_coeff_coef - 1).mul_(div_coeff_coef)
			if Basis:
				mul_coeff_basis = 2 ** (bits_basis - 1 - int_bits_basis )
				div_coeff_basis = 2 ** (int_bits_basis - bits_basis + 1)
				max_coeff_basis = 2 ** (bits_basis - 1)
				self.basis = self.basis.mul_(mul_coeff_basis).floor_().clamp_(-max_coeff_basis - 1, max_coeff_basis - 1).mul_(div_coeff_basis)

		# with torch.no_grad():
		# 	kmeans = KMeans(n_clusters = 2 ** bits)
		# 	coefs = self.coefs.detach().cpu().numpy()
		# 	coefs = coefs[coefs!=0].reshape((-1, 1))
		# 	kmeans.fit(coefs)
		# 	fake_quant_val  = kmeans.cluster_centers_[kmeans.labels_]
		# 	self.coefs[self.coefs!=0] = torch.Tensor(fake_quant_val.flatten()).cuda()
	def rounding(self, bits, int_bits = 0, thresh = 3):
		'''
		Rounding the coefficients to the nearest 
		'''
		codebook = np.flip([ 2 ** x for x in range(-(bits-int_bits), int_bits)])
		#print(codebook)
		candidates = self.coefs[self.coefs!=0].detach().cpu().numpy()

		self.indicator = np.zeros((candidates.shape[0], bits))

		new_coefs = np.zeros_like(candidates)

		#print(candidates)
		code_id = 0
		for code in codebook:
			self.indicator[np.abs(candidates/code)>= 1, code_id] = 1

			mask = self.indicator[:, code_id]==1
			mask[np.sum(self.indicator[:, :code_id], axis=1) > thresh] = 0

			candidates[mask] -= np.sign(candidates[mask]) * code 
			new_coefs[mask] += np.sign(candidates[mask]) * code 
			code_id +=1

		#print(new_coefs)
		#print("Rounding MSE: ", np.sum(candidates ** 2))
		with torch.no_grad():
			self.coefs[self.coefs!=0] = torch.Tensor(new_coefs).cuda()
	def compute_4col_loss(self):
	    num_cols = int((self.in_channels * self.num_basis - 1) // 4 + 1)
	    if num_cols * 4 == self.in_channels * self.num_basis:
	        loss = torch.norm(torch.norm(self.coefs.view(self.out_channels, num_cols, 4),p=1, dim=2), p=2)
	    else:
	        coef_view = self.coefs.view(self.out_channels, self.in_channels * self.num_basis)
	        partial_l1 = torch.norm(coef_view[:, :(num_cols-1) * 4].view(self.out_channels, num_cols-1, 4) , p=1, dim=2)
	        residual_l1= torch.norm(coef_view[:, (num_cols-1) * 4:].view(self.out_channels, 1, -1) , p=1, dim=2)
	        loss = torch.norm(torch.cat((partial_l1, residual_l1), 1), p=1)
	    return loss.cuda().view(1)


	def prune_4col_loss(self, s=1.0):
	    with torch.no_grad():
	        num_cols = int((self.in_channels * self.num_basis - 1) // 4 + 1)
	        if num_cols * 4 == self.in_channels * self.num_basis:
	            coef_view = self.coefs.view(-1, 4)
	            norms = torch.norm(self.coefs.view(self.out_channels, num_cols, 4),p=1, dim=2)
	            threshold = torch.std(norms) * s
	            mask = (norms<=threshold).flatten()
	            coef_view[mask, :] = 0
	        else:
	            coef_view = self.coefs.view(self.out_channels, self.in_channels * self.num_basis)
	            partial_norms = torch.norm(coef_view[:, :(num_cols-1) * 4].view(self.out_channels, num_cols-1, 4) , p=1, dim=2)
	            residual_norms= torch.norm(coef_view[:, (num_cols-1) * 4:].view(self.out_channels, 1, -1) , p=1, dim=2)
	            threshold = torch.std(torch.cat((partial_norms, residual_norms), 1)) * s
	             
	            partial_mask = (partial_norms<=threshold).flatten()
	            partial = coef_view[:, :(num_cols-1) * 4].reshape(self.out_channels *(num_cols-1), 4)
	            partial[partial_mask, :] = 0
	            coef_view[:, :(num_cols-1) * 4] = partial.reshape(self.out_channels, (num_cols-1) * 4)
	            
	            residual_mask = (residual_norms<=threshold).flatten()
	            residual = coef_view[:, (num_cols-1) * 4:].reshape(self.out_channels, -1)
	            residual[residual_mask, :] = 0
	            coef_view[:, (num_cols-1) * 4:] = residual
	        

	def compute_p2_loss(self, bits, int_bits = 0, thresh=0):
		# codebook = [ 2 ** x for x in range(-(bits-int_bits), int_bits)]
		# candidates = self.coefs[self.coefs!=0]
		# pre_codebook = torch.ones_like(candidates, dtype=torch.bool)
		# total_loss = 0
		# for weight in codebook:
		# 	new_codebook = (candidates < weight)
		# 	round_hit = pre_codebook & new_codebook
		# 	candidates[round_hit] = weight
		# 	pre_codebook = (candidates > weight)
		# total_loss += torch.sum((self.coefs[self.coefs!=0] - candidates) ** 2)
		# return total_loss

		codebook = np.flip([ 2 ** x for x in range(-(bits-int_bits), int_bits)])
		candidates = self.coefs[self.coefs!=0].detach().clone()

		indicator = torch.zeros((candidates.shape[0], bits)).cuda()
		new_coefs = torch.zeros_like(candidates).cuda()

		#print(candidates)
		code_id = 0
		for code in codebook:
			indicator[torch.abs(candidates/code)>= 1, code_id] = 1

			mask = indicator[:, code_id]==1
			mask[torch.sum(indicator[:, :code_id], axis=1) > thresh] = 0

			candidates[mask] -= torch.sign(candidates[mask]) * code 
			new_coefs[mask] += torch.sign(candidates[mask]) * code 
			code_id +=1

		#print(new_coefs)
		#print("Rounding MSE: ", torch.sum(candidates ** 2))

		#print(torch.sum((self.coefs[self.coefs!=0] - new_coefs) ** 2).detach())

		return torch.sum((self.coefs[self.coefs!=0] - new_coefs) ** 2) 

	def compute_group_lasso(self, chunk_size = 32):
		layer_loss = torch.zeros(1).cuda()

		num_chunks = int((self.out_channels - 1) // chunk_size + 1)
		if num_chunks * chunk_size == self.out_channels:
			loss = torch.norm(torch.norm(self.coefs.view(num_chunks, chunk_size, self.in_channels * self.num_basis), p=1, dim=1), p=2)
		else:
			coef_view = self.coefs.view(self.out_channels, self.in_channels * self.num_basis)
			partial_l1 = torch.norm(coef_view[:(num_chunks-1)*chunk_size, :].view(num_chunks-1, chunk_size, -1), p=1,
									dim=1)
			residual_l1 = torch.norm(coef_view[(num_chunks-1)*chunk_size:, :].view(1, -1, self.in_channels * self.num_basis), p=1, dim=1)
			loss = torch.norm(torch.cat((partial_l1, residual_l1), 0), p=2)
		return loss.cuda().view(1)

	def prune_by_chunk(self, chunk_size=32, s=1.0):
		with torch.no_grad():
			num_chunks = int((self.out_channels - 1) // chunk_size + 1)
			if num_chunks * chunk_size == self.out_channels:
				coef_view = self.coefs.view(num_chunks, chunk_size, self.in_channels * self.num_basis)\
							.transpose(1, 2).view(-1, chunk_size)
				norms = torch.norm(coef_view, p=1, dim=1)
				threshold = torch.std(norms) * s
				mask = (norms <= threshold).flatten()
				coef_view[mask, :] = 0
				self.coefs.data = coef_view.reshape(num_chunks, self.in_channels *  self.num_basis ).transpose(1,2).view(self.out_channels * self.in_channels, self.num_basis)
			else:
				coef_view = self.coefs.view(self.out_channels, self.in_channels * self.num_basis)
				partial_norms = torch.norm(coef_view[:(num_chunks-1)*chunk_size, :].view(num_chunks-1, chunk_size, -1),
										   p=1, dim=1)
				residual_norms = torch.norm(coef_view[(num_chunks-1)*chunk_size:, :].view(1, -1, self.in_channels * self.num_basis), p=1,
											dim=1)
				threshold = torch.std(torch.cat((partial_norms, residual_norms), 0)) * s

				partial_mask = (partial_norms <= threshold).flatten()
				partial = coef_view[:(num_chunks-1)*chunk_size, :].view(num_chunks-1, chunk_size, -1).transpose(1, 2).reshape(-1, chunk_size)
				partial[partial_mask, :] = 0

				coef_view[:(num_chunks-1)*chunk_size, :] = partial.reshape((num_chunks-1)*chunk_size, -1)

				residual_mask = (residual_norms <= threshold).flatten()
				residual = coef_view[(num_chunks-1)*chunk_size:, :].view(-1, self.in_channels * self.num_basis).transpose(0, 1)
				residual[residual_mask, :] = 0
				coef_view[(num_chunks-1)*chunk_size:, :] = residual.transpose(1,0)

	def compute_chunk_score(self, chunk_size=32):
		score  = 0
		with torch.no_grad():
			num_chunks = int((self.out_channels - 1) // chunk_size + 1)
			nz_chunk = 0
			if num_chunks * chunk_size == self.out_channels:
				coef_view = self.coefs.view(num_chunks, chunk_size, self.in_channels * self.num_basis)\
							.transpose(1, 2).view(-1, chunk_size)
				norms = torch.norm(coef_view, p=1, dim=1)
				threshold = torch.std(norms)
				mask = (norms <= threshold).flatten()
				nz_chunk = torch.sum(mask)
				coef_view[mask, :] = 0
			else:
				coef_view = self.coefs.view(self.out_channels, self.in_channels * self.num_basis)
				partial_norms = torch.norm(coef_view[:(num_chunks-1)*chunk_size, :].view(num_chunks-1, chunk_size, -1),
										   p=1, dim=1)
				residual_norms = torch.norm(coef_view[(num_chunks-1)*chunk_size:, :].view(1, -1, self.in_channels * self.num_basis), p=1,
											dim=1)
				threshold = torch.std(torch.cat((partial_norms, residual_norms), 0))

				partial_mask = (partial_norms <= threshold).flatten()
				residual_mask = (residual_norms <= threshold).flatten()
				nz_chunk = torch.sum(partial_mask) + torch.sum(residual_mask)

			score = nz_chunk / (num_chunks * self.in_channels * self.num_basis)

		return score.detach().item()

	def compute_chunk_score_post(self, chunk_size, tol=0.0):
		last_chunk = (self.out_channels) % chunk_size
		n_chunks = (self.out_channels) // chunk_size + (last_chunk != 0)

		coef_mat = self.coefs.view((self.out_channels, -1))

		total_score = 0.
		total_nz_chunks = 0

		for chunk_idx in range(n_chunks):
			if chunk_idx == n_chunks - 1 and last_chunk != 0:
				current_chunk = coef_mat[chunk_idx * chunk_size:, : ].detach()
			else:
				current_chunk = coef_mat[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, : ].detach()
			current_chunk[torch.abs(current_chunk)<tol] = 0.
			tmp_sum = torch.sum(torch.abs(current_chunk), dim=0)
			#print(tmp_sum)
			total_nz_chunks += torch.sum(tmp_sum!=0).cpu().item()
			local_score = torch.sum(tmp_sum==0) / float(self.num_basis * self.in_channels)
			total_score += local_score
		return total_score.cpu().item() / n_chunks, total_nz_chunks, n_chunks * self.num_basis * self.in_channels

	def forward(self, x):
		#with torch.no_grad():
		#	thresh = torch.std(self.coefs)
		#	self.coefs[torch.abs(self.coefs)<thresh] = 0
		#	self.coefs[torch.abs(self.coefs)>=thresh] = 1
		#self.coefs.data = self.coefs.transpose(0,1).reshape(self.num_basis * self.out_channels, self.in_channels) * self.scales.view(-1, 1)
		#self.coefs.data = self.coefs.view(self.num_basis, self.out_channels * self.in_channels).transpose(0, 1)
                
		#true_weight = torch.mm(self.coefs, self.basis).view((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
		true_weight = torch.mm(self.coefs.reshape(self.num_basis, self.out_channels * self.in_channels).transpose(0, 1), self.basis).view((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
		out = F.conv2d(x, true_weight, self.bias, self.stride, self.padding, self.dilation)

		"""
		weights = true_weight.detach().cpu().numpy()
		weights[weights > 1e-6] = 1
		weights[weights < -1e16] = 1
		weights[weights != 1] = 0
		weights = weights.astype(int)
		print("DecomposedConv2D_" + str(common.COUNT-1))
		print(np.shape(weights))
		weights = np.reshape(weights, (np.shape(weights)[0], -1),'C')
                
		#PATH = '/root/reproduce/cambriconx/data/vgg16/cifar10/'
		#PATH = '/root/reproduce/cambriconx/data/mobilenetv2/cifar10/'
                
		pd.DataFrame(weights).to_csv(common.PATH + 'DecomposedConv2D_' + str(common.COUNT-1) + '.csv')
		exportIFM(x, True, True, common.PATH + "act" + str(common.COUNT-1) + ".csv")
		common.COUNT += 1
		#"""                

		return out

	def forward_oc(self, x):
		print("--OC--")
		coefs = torch.bmm(self.ic_vecs, self.k_vecs)
		true_weight = torch.mm(coefs.view(self.out_channels * self.in_channels, self.num_basis), self.basis).view(
			(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
		out = F.conv2d(x, true_weight, self.bias, self.stride, self.padding, self.dilation)

		return out

	def forward_train(self, x):
		pass

	def forward_pre(self, x):
		pass
	# 	#To get the essential runtime information
	# 	true_weight = torch.mm(self.coefs, self.basis).view((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
	# 	out = F.conv2d(x, true_weight, self.bias, self.stride, self.padding, self.dilation)
	#
	# 	self.out_w = out.shape[3]
	# 	self.out_h = out.shape[2]
	#
	# 	return out

	def extra_repr(self):
		return 'in1_channels={}, out_channels={}, num_basis = {}, bias={}'.format(
			self.in_channels, self.out_channels, self.num_basis, self.bias is not None)

	def forward_test(self, x):
		pass
	# 	#Efficient Implementation
	# 	# 1          1 1 1 1
	# 	# 2   ---->  2 2 2 2
	# 	# 3          3 3 3 3
	# 	basis_kernel = self.basis.repeat((1, self.in_channels)).view((self.num_basis*self.in_channels, 1, self.kernel_size[0], self.kernel_size[1]))
	# 	#w = ((x.shape[2] + self.padding[0]) - self.kernel_size[0]//2) //self.stride[0]
	# 	mid_fm = conv2d_depthwise(x.repeat((1,self.num_basis, 1, 1)), basis_kernel, self.bias, self.stride, self.padding, self.dilation)
	#
	# 	del basis_kernel
	#
	# 	#Temp Variable
	# 	sIFM = torch.zeros((self.out_h * self.out_w * self.values.shape[0],), dtype=torch.float32).cuda()
	# 	out = torch.zeros((x.shape[0] * self.out_channels * self.out_h * self.out_w,), dtype=torch.float32).cuda()
    #
	# 	#self.scaling(grid=((self.values.shape[0] + 1024) // 1024, self.out_h, self.out_w), block = (1024,1,1),
	# 	#			 args=[sIFM.data_ptr(), mid_fm.data_ptr(), self.values.data_ptr(), self.col_idx.data_ptr(), self.row_idx.data_ptr()],
	# 	#			 stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
	# 	del mid_fm
	# 	#self.accumulate(grid=(self.out_channels, (self.out_h + 32) // 32, (self.out_w + 32) // 32), block = (1,32,32),
	# 	#			 args=[out.data_ptr(), sIFM.data_ptr(), self.acc_ptr.data_ptr() ],
	# 	#			 stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
	# 	#mid_fm = F.conv2d(x.repeat((1,self.num_basis, 1, 1)), basis_kernel, self.bias, self.stride, self.padding, self.dilation, groups = self.num_basis * self.in_channels)
	# 	#out = F.conv2d(mid_fm, self.coefs.view(self.out_channels, self.in_channels * self.num_basis, 1, 1), stride=1, padding=0, dilation=1)
	# 	#out = self.coefs.mm(conv2d_depthwise(x.repeat((1,self.num_basis, 1, 1)), basis_kernel, self.bias, self.stride, self.padding, self.dilation)\
	# 	#	.view((-1, w * w))).view((1, self.out_channels, w, w))
	# 	del sIFM
	#
	# 	return out.view((x.shape[0], self.out_channels ,self.out_h, self.out_w))
