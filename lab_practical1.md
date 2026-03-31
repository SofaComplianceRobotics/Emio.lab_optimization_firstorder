# First-Order Optimization: Model Learning

::: highlight
##### Overview

The goal of this lab is to learn the inverse kinematics of Emio (calculating the required motor angles for a desired end-effector position), using a multilayer perceptron (MLP) to model the mapping from end-effector position to motor angles.

You will build, train, and evaluate the MLP using PyTorch, and in the end of the lab, you will calibrate an alternative, parametric model. 

:::

:::: collapse Install Dependencies

We are going to need third-parties libraries for this lab.

Click the button below to install them:
#python-button("-m pip install --target 'assets/labs/Practical1/modules/site-packages' -r 'assets/labs/Practical1/requirements.txt'")

::::

<a id="datasets"></a>
:::: collapse Datasets
## Datasets 

The datasets used in this lab are in CSV files containing the motors angles and the corresponding end-effector positions of Emio. The datasets are located in the `data/results` folder. Both datasets have the following fields:
- the four motors angles _m0_, _m1_, _m2_ and _m3_
- the 3D position of the effector _pos_


### Simulation

Two datasets, created in simulation, are available:
- `blueleg_beam_cube.csv`:
- `blueleg_beam_sphere.csv`:

They have been generated using the SOFA simulation of Emio, with the script `dataset_generation.py`.

You can take a look at `blueleg_beam_cube.csv`: 
#open-button(file="assets/labs/Practical1/data/results/blueleg_beam_cube.csv")

### Real Robot

Equivalent datasets were recorded on the Emio robot:
- `blueleg_beam_real_cube2197.csv`
- `blueleg_beam_real_sphere1018.csv`

These datasets were created by tracking the robot's tool center point (TCP) position with a _Polhemus_ magnetic tracker. These datasets have an extra column `Real Position` with the recorded tracked position.

You can take a look at `blueleg_beam_real_cube2197.csv`: 
#open-button(file="assets/labs/Practical1/data/results/blueleg_beam_real_cube2197.csv")

::::

:::: collapse Create MLP Model

### Create MLP Model

You will use a multilayer perceptron (MLP) with two hidden layers of 128 neurons each. The input layer will have 3 neurons (the x, y, z coordinates of the end-effector position) and the output layer will have 4 neurons (the 4 motors angles).

The activation function used in the hidden layers is the sigmoid function and there is no activation function in the output layer.

::: exercise
**Exercise 1:**
In the file `modules/pytorch_mlp.py`, complete the code to create a PyTorch MLP 
with 2 linear layers of 128 neurons each (`nn.Linear`), and a sigmoid activation function at the hidden layers (`nn.Sigmoid`).

#open-button(file="assets/labs/Practical1/modules/pytorch_mlp.py")

:::

::::

:::: collapse Train MLP Model
### Train MLP Model 

To train your model, you will run the provided `train_model.py` script. 
The script will preprocess the data, build the MLP, train it, and save the trained model to the specified location.

::: exercise
**Exercise 2:**

1. In `modules/pytorch_mlp.py`, finish implementing the training loop. As loss, use the mean-square error `nn.MSELoss()`. As solver, you can use the Adam algorithm `optimizer = optim.Adam(self.model.parameters())`
#open-button(file="assets/labs/Practical1/modules/pytorch_mlp.py")

2. Train the model, using the `train_model.py`: 

    #python-button(file="assets/labs/Practical1/train_model.py", pyargs=["--model-type", "pytorch", "--dataset-path",  "assets/labs/Practical1/data/results/blueleg_beam_cube.csv"])

3. Inspect the convergence. If necessary, tune the parameters of Adam for better results. 

:::
::::

:::::: collapse Evaluate MLP Model
### Evaluate MLP Model

First, we can do a statistical evaluation. We evaluate the performance of the trained dataset on other datasets.  

::::: exercise
**Exercise 3:**

Evaluate the learned model:

Try with each by each of the four [datasets](#datasets). 

:::: select eval_pytorch_dataset 
::: option assets/labs/Practical1/data/results/blueleg_beam_cube.csv
::: option blueleg_beam_sphere.csv
::: option blueleg_beam_real_cube2197.csv
::: option blueleg_beam_real_sphere1018.csv
::::

Comment in your report. On what dataset does the model perform best? On which one does it perform worst? Can you explain the observed behavior? 

#python-button(file="assets/labs/Practical1/evaluate_model.py", pyargs=["--model-type", "pytorch", "--dataset-path",  "eval_pytorch_dataset", "--model-path", "assets/labs/Practical1/data/results/blueleg_beam_cube.pth"])


:::::

Finally, you can use your model to control the robot. The scene `sofa_sim.py` is already set up to use your trained model. You just need to specify the path to your model `.pth` file in the scene:
#input("eval_pytorch_model_path", "Path to the model pth file", "assets/labs/Practical1/data/results/blueleg_beam_cube.pth")

The effector will then move to the different targets sampled along the sphere or cube, as shown below:

![](assets/labs/Practical1/data/images/evaluation_sphere.png)


::: exercise

**Exercise 4:**

Run the sofa simulation and observe how the robot moves to the prescribed points. Describe the behavior in your report. 

#runsofa-button("assets/labs/Practical1/sofa_sim.py", "eval_pytorch_model_path", "sphere", "0.1")

:::

After successfully completing Exercise 4 and showing the working simulation to your teaching crew, you may continue with Exercise 5. 

::: exercise

**Exercise 5:**

Run the above script on the real robot. Describe the observed behavior in your report. 

:::

::::::

:::: collapse Parametric Model Learning 

Learning inverse kinematics with a deep neural network is one way to do things, but certainly not the only and possibly not the optimal way. In the next practical, we 
will solve inverse kinematics using a model-based way. However, to get good performance, we will need accurate models. We can use physical principles to setup good models but there are always some parameters that need to be tuned. We can learn these parameters using collected data. This is called **calibration** or parametric model learning.

::: exercise
**Exercise 6:**

- In `train_model.py`, there is an option to use `calibrated` instead of `pytorch`. Inspect the code for the proposed calibration and comment on the implementation. In particular, what principle is being used here to calibrate the Young modulus?
    #open-button(file="assets/labs/Practical1/train_model.py")

- Go to `train_model.py` and make sure the default variable is set to calibrated: `DEFAULT="calibrated"`.
    #open-button(file="assets/labs/Practical1/train_model.py")

- By clicking the below button, you run `train_model.py` using the calibrated option. Observe the convergence behavior. Do you understand why the algorithm behaves the way it does? 

#python-button(file="assets/labs/Practical1/train_model.py" pyargs=["--dataset-path", "assets/labs/Practical1/data/results/blueleg_beam_sphere.csv"])


::: 

In Practical 2, we will fix the above behavior and use the calibrated model in our inverse kinematics pipeline, and you can then properly compare this model-based approach with the deep-learning approach. 

::::


### Appendix 

:::::: collapse Dataset Generation

::::: exercise
**Generation SOFA Scene:**

You can generate your own dataset using this scene.
This will generate a dataset into the _data/results_ folder.

:::: select dataset_shape 
::: option sphere
::: option cube
::::

#input("dataset_ratio", "Ratio to sample (the higher the coarser)", "0.08")

#runsofa-button("assets/labs/Practical1/dataset_generation.py", "dataset_shape", "dataset_ratio")

<br>

Here is is an excerpt of the _blueleg_beam_sphere.csv_ dataset file that comes with this lab:

```text
# extended ;1
# legs ;['blueleg']
# legs model ;['beam']
# legs young modulus ;[35000.]
# legs poisson ratio ;[0.45]
# legs position on motor ;['counterclockwisedown', 'clockwisedown', 'counterclockwisedown', 'clockwisedown']
# connector ;bluepart
# connector type ;rigid
Effector position;Motor angle
[-39.96175515 -90.41789743 -39.96175525];[-0.14670205865712832, 0.14670207392254797, 2.43823807942873, -2.438238056118855]
[-39.95720099 -90.4415037  -31.95609913];[0.1329811350050557, 0.13624487007045172, 2.29165178728331, -2.488099187824528]
[-39.95397373 -90.45505537 -23.95436099];[0.4217800556565714, 0.13599500085968805, 2.113863614076125, -2.5101582582004642]
[-39.9514583  -90.46332739 -15.96017202];[0.7233308263521361, 0.14326494422921077, 1.8979428979904553, -2.5098560718291005]
[-39.95029449 -90.46182801  -7.97640971];[1.0359369002803307, 0.15246389567464924, 1.640800631571854, -2.4992699487352867]
[-3.99504339e+01 -9.04556845e+01 -8.41130293e-05];[1.3485569542409783, 0.1566254899703859, 1.3478513803217718, -2.4934454150763674]
```

:::::

::::::
