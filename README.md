# MBRL2 (Model Based Regularization for Bayesian Reinforcement Learning)
Code for the paper - "Meta Reinforcement Learning with Finite Training Tasks - a Density Estimation Approach". 

This code is based on the open-sourced VariBad repository of Zintgraf et al -
https://github.com/lmzintgraf/varibad.
For general overview of the repository, requirements and hyperparameters we refer the reader 
to the original VariBad repository.

# Dream Environments Options
Besides the config options introduces in the VariBad repo, 
1. **env_num_train_goals, env_num_eval_goals** - number of training and evaluation environmnets
2. **num_dream_envs** - number of dream environments processes
3. **use_kde, use_mixup** - use KDE to sample new latents, if false we use the learned Prior
4. **use_mixup** - use the mixup technique to sample new latents instead of regular KDE
5. **delay_dream** - number of iterations to delay the initialization of the dream environments by
6. **update_kde_interval** - iterations interval for the KDE updates
7. **kde_from_train** - create KDE using an oracle policy
8. **kde_from_running_latents** - use a latent pool, gathered along the training for the dream environments estimation
9. **freeze_vae** - don't train the VAE (only the policy)
10. **delayed_freeze** - stop the VAE training after given number of iterations
11. **train_vae_on_dream** - train the VAE to reconstruct reward over the dream environments
12. **clone_dream_vae** - use a different vae for the dream environments

# Reproducing Results
In order to reproduce the results shown in the paper:
1. For the 20 real training environments and 4 dream environment:
   ```sh
   python main.py --exp_name 20_train_4_kde_dream --env_type pointrobot_varibad\
                  --env_num_train_goals 20 --num_dream_envs 4
   ```
   
2. For the 30 real training environments and 6 dream environment:
   ```sh
   python main.py --exp_name 30_train_6_kde_dream --env_type pointrobot_varibad \
                  --env_num_train_goals 30 --num_dream_envs 6
   ```

In order to use Mixup dream environments instead of the KDE, add the --use_mixup flag.  

