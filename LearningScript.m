close all
clear;
clc;

rng default %For reproducibility

%% Load and organize data

%Data is organized with each column corresponding to a numerical features, and each row corresponding to an
%instance
%Labels are a one dimensional column vector

trainingData = 0;
trainingLabels = 0;

validationData = 0;
validationLabels = 0;


%% Run NeuroEvolution

popSize = 500; %Population size to use
numGen = 500; %Number of generations to run for

ne = Evolver(popSize, trainingData, trainingLabels, validationData, validationLabels);
ne.evolve(numGen);

%% Construct neural network object and plot validation data scatter plot

%Get best candidate
bestCan = ne.BestCandidate;

%Construct neural network
nn = NeuralNetwork(bestCan.InDim, bestCan.HidDim, bestCan.OutDim, bestCan.Wi, bestCan.Bi, bestCan.Wo, bestCan.Bo);

%Get predicted validation outputs
valOut = nn.outputsForDataset(validationData);

figure
hold on
plot(validationLabels,valOut,'*')
xlabel('Truth')
ylabel('Predicted')
hold off



























