# MOTOKEN: Interpreting Motion Tokens in Video Diffusion Models

Bar Alon · Eldad Cohen · Ariel Shaulov  
Tel Aviv University · baralon1@mail.tau.ac.il · eldadc1@mail.tau.ac.il · arielshaulov@mail.tau.ac.il

---

## 🚀 Project Overview

**MOTOKEN** investigates how state-of-the-art text-to-video diffusion models encode motion dynamics in their internal token representations. We analyze the latent tokens of CogVideoX-5B, propose a lightweight **token-mixture embedding** mechanism to steer motion, and develop a composite training objective that balances:

1. **Conditional prediction loss**: match placeholder-conditioned noise residuals to concept-conditioned ones  
2. **Multi-scale motion loss**: enforce temporal coherence across several frame-difference scales  
3. **Embedding regularization**: encourage the learned embedding to remain semantically close to the target concept  

Despite partial success on simple “walking” prompts, our work uncovers tight coupling between appearance and motion tokens, highlights limitations of current backbones, and suggests future directions for explicit motion disentanglement.

---

## 📦 Repository Structure

```
├── data/                      # (Optional) Downloaded or pre-generated video clips
│   ├── dog_walking/               # 100 synthetic “dog walking” videos
│   ├── man_walking/               # 100 synthetic “man walking” videos
│   └── ...                        # other motion classes (running, painting, etc.)
├── models/                        # Definitions for Net, LanguageBind interface, similarity model
│   ├── languagebind/      
│   └── model_zoo.py
├── scripts/               
│   ├── cog_inference.py           # Generate videos using cog
│   ├── create_dataset.py          # Create 100 videos for the dataset
│   ├── model_and_losses.py        # Create 100 videos for the dataset
│   ├── utils.py
│   └── one_step_reconstruction.py # The main train loop
├── environment.yml        
├── README.md              # This file
└── nlp_project.pdf        # Full project report
```

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/motoken.git
   cd motoken
   ```

2. Create and activate the Conda environment from environment.yml:
   ```bash
   conda env create -f environment.yml
   conda activate motoken
   ```

---

## 🏋️‍♂️ Training

Train the token-mixture embedding end-to-end on your synthetic dataset:

```bash
python python -u motoken/scripts/one_step_reconstruction.py --prompt="A man walking" \
                                                            --concept=" walking" \
                                                            --pretrained_model_name_or_path="THUDM/CogVideoX-5B" \
                                                            --placeholder_token="<>" \
                                                            --train_batch_size=2 \
                                                            --validation_prompt="A man <>"  \
                                                            --train_data_dir="./motoken/data/A_man_walking/train" \
                                                            --validation_data_dir="./motoken/data/A_man_walking/eval" \
                                                            --output_dir="./motoken/output/man_walking_tokens_removed_lr_${LR}" \
                                                            --dictionary_size=30000 \
                                                            --num_explanation_tokens=50 \
                                                            --validation_steps=50 \
                                                            --learning_rate=$LR \
                                                            --max_train_steps=500 \
                                                            --seed=1024 \
                                                            --validation_seed=1024 \
                                                            --cond_press_coeff=10 \
                                                            --motion_coeff=0.01 \
                                                            --num_validation_videos=3 \
                                                            --remove_concept_tokens \
                                                            --gradient_checkpointing
```



## 📈 Results

- **Learned mixtures**: “hiking” token dominates for both dog and man walking (≈ 0.4–0.6 weight).  
- **Qualitative**: intermittent background/camera-angle consistency between “walking” and related tokens; full disentanglement remains elusive.  
- **Ablation**: multi-scale motion loss critical for silhouette coherence; embedding regularizer necessary to induce actual walking motion.  
- **Sanity check**: re-introducing “walking” token yields α≃1 within a few iterations, confirming optimization correctness.

Refer to `nlp_project.pdf` for full figures, tables, and in-depth discussion.

---

## 🔮 Future Work

1. **Motion-capable generator**: replace CogVideoX-5B with a backbone trained on concise action prompts.  
2. **Complex and composite motions**: extend beyond simple walking to richer action classes.  
3. **Multi-step embedding optimization**: accumulate gradient signals over several timesteps for robust α convergence.  


## 📫 Contact

For questions or collaborations, please reach out to:

- Bar Alon: baralon1@mail.tau.ac.il  
- Eldad Cohen: eldadc1@mail.tau.ac.il  
- Ariel Shaulov: arielshaulov@mail.tau.ac.il  

Happy exploring motion in diffusion! 🚀
