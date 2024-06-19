This repository contains a Kubeflow-based MLops app to detect spam message, built upon a data pipeline for extraction, preprocessing, training, and model testing.

Kubeflow is a powerful tool designed to simplify and streamline machine learning operations (MLOps) on Kubernetes.

# Why Kubeflow
## Scalability
Kubeflow runs on Kubernetes, leveraging its ability to scale computational resources up and down as needed. This allows for efficient handling of large datasets and complex models. Plus, Kubeflow supports distributed training of machine learning models, making it easier to train large models on clusters of machines.

## Portability
Kubeflow is designed to be platform-agnostic (i.e. cross platform), meaning it can run on any cloud provider that supports Kubernetes (e.g., GCP, AWS, Azure) as well as on-premises clusters. 

## Automation
Kubeflow supports scheduling of pipeline runs, allowing for automatic retraining and evaluation of models at regular intervals. Also, built-in support for hyperparameter tuning with tools like Katib helps automate the optimization of model parameters.


-----------------

Note: please ensure you have installed Kubeflow Pipeline (KFP) package version 2.7.0, as significant changes in each version may impact code functionality. To install version 2.7.0, run the following command
```
pip install kfp==2.7.0
```
-----------------
# Prerequisites for Working with Kubernetes

### Container Runtime
Example: Docker

### Kubernetes Command-Line Interface
Tool: kubectl

### Kubernetes Cluster Management Tool:
Example: Kubeadm

### Network Plugin:
Examples: Calico, Flannel, Weave Net

### Container Registry:
Examples: Docker Hub, Google Container Registry


-----------------
# Key Components of Kubeflow

### Kubernetes Cluster

Description: A Kubernetes cluster is a set of nodes that run containerized applications managed by Kubernetes. It consists of a master node (control plane) and worker nodes where applications run.

Example: Minikube

### Kubernetes Command-Line Interface

Description: kubectl is the command-line tool used to interact with Kubernetes clusters. It allows you to deploy applications, inspect and manage cluster resources, and view logs.

Example: kubectl, the standard CLI tool provided by Kubernetes.

### Configuration Management Tool:

Description: Tools like Kustomize manage Kubernetes configurations. They allow you to customize Kubernetes YAML configurations without using templates, making it easier to manage different environments (e.g., development, staging, production).

Example: Kustomize, a tool for customizing Kubernetes resources declaratively.


### Container Runtime:

Description: A container runtime is the software that runs containers. It is responsible for managing container lifecycle, including starting, stopping, and managing container images.

Example: Docker, a widely used container runtime that manages containerized applications

-----------------

# How to deploy Kubeflow app
When YAML file of app is compiled, you can either deploy it on local or cloud based host.

## Local Deployment
You can deploy Kubeflow locally using tools like Minikube or Kind (Kubernetes in Docker). This approach is great for development and testing purposes.

Minikube: Minikube is a tool that runs a single-node Kubernetes cluster on your local machine.

Kind: Kind runs Kubernetes clusters in Docker containers.

## Cloud Deployment
Kubeflow can be deployed on various cloud platforms, including Google Cloud Platform (GCP), Amazon Web Services (AWS), Microsoft Azure.




