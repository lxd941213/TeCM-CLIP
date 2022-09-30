import torch
from torch.utils.data import Dataset
import numpy as np
import clip


class LatentsDataset(Dataset):

	def __init__(self, latents, opts):
		self.latents = latents
		self.opts = opts

		with open(self.opts.description, "r") as fd:   # facestyle_description
			self.description_list = fd.read().splitlines()

		self.styles_list = [single_description for single_description in self.description_list]

	def style_from_image_text(self, index):
		selected_style_description = np.random.choice(self.styles_list)
		style_text_embedding = torch.cat([clip.tokenize(selected_style_description)])[0]

		return self.latents[index], style_text_embedding

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):
		return self.style_from_image_text(index)


class LatentsDataset_inference(Dataset):

	def __init__(self, latents, opts):
		self.latents = latents
		self.opts = opts


	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):

		return self.latents[index]

