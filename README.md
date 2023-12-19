This repository includes part of the code from HyperTagging project (proposed and studied by myself) for the event level embedding using unsupervised learning 
trained with continues contrastive loss or regularisation loss in hyperbolic space.  

Abstract for HyperTagging:
In analyses at Belle II, it is often helpful to reconstruct the whole decay process of each electron-positron collision event using the information collected from detectors. 
The reconstruction is composed of several steps which require manual configurations and suffers from high uncertainty as well as low efficiency. In this project, I am developing 
a software with the aim to reconstruct B decays at Belle II automatically with both high efficiency and high accuracy. The well-trained models should be tolerant to rare decays 
that have very small branching ratio or are even unseen during the training. To ensure high performance, the project is separated into several stages: particle level embedding, 
event level embedding and decay reconstruction. Inspired by the recent achievements in computer science, transformers and hyperbolic embedding are employed as building blocks 
with pre-training-fine-tuning framework, contrastive metric learning and knowledge transfer serving as training tools.

An open presentation with slides on this project can be found at https://indico.ihep.ac.cn/event/19430/
