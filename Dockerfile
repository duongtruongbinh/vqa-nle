# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# (Optional) Specify a command to run when the container starts
# For example, to start a training run:
# CMD ["python", "src/training/run_grpo.py", "--config", "configs/grpo_config_qwen.yaml"]
