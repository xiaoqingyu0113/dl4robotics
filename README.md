# dl4robotics
Robot learning in Isaac sim with wheeled tennis robot

### dependencies

1. Omniverse Isaac Sim
2. stable-baseline3
3. openai gym

### test agent with action inputs
1. open and edit the file test_env.py
2. change the inputs of actions to your desired.` Note: all the inputs are scaled to [-1,1]`.
3. run the test
   ```bash
    PYTHONPATH  test_env.py
    ```

###  training

1. run the test training
   ```bash
    PYTHONPATH  test_train.py
    ```
2. run the training with stable-baseline3
   ```bash
    PYTHONPATH  train.py
    ```
3. visualize the training result with stable-baseline3. `(need to edit and load the correct model in the validate.py)`
   ```bash
    PYTHONPATH  validate.py
    ```