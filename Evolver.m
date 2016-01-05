%Evolves a sparsely connected, single hidden layer, neural network for
%unrestricted MSE regression

classdef Evolver < handle
    
    properties
        
        NumGenerations; %Counts the number of generations executed
        PopSize; %The maximum size of the population
        
        %Data used for fitness evalution
        TrainingData; %Features along columns, instances along rows
        TrainingLabels; %Single dimensional real valued, instances along rows
        
        %Data used to show generalization performance to user, not used in
        %candidate selection
        ValidationData; %Features along columns, instances along rows
        ValidationLabels; %Single dimensional real valued, instances along rows
        
        Population; %Stores the current population of candidates
        
        BestCandidate; %The candidate with best fitness so far, across all generations, on training data
        BestTrainingFitness; %The corresponding fitness of the best candidate
        BestValidationFitness; %The corresponding fitness on the validation data
        
        InDim; %The input dimensionality of the network (input layer size / number of features)
        OutDim; %The number of simultaneous regression targets (hard coded to 1 in this version)
        
    end
    
    methods
        
        %Constructor: inputs parameters are defined above by their
        %corresponding property
        function obj = Evolver(popSize, trainingData, trainingLabels, validationData, validationLabels)
            
            %Store instance variables
            obj.PopSize = popSize;
            obj.InDim = size(trainingData,2);
            obj.OutDim = 1;
            obj.TrainingData = trainingData;
            obj.TrainingLabels = trainingLabels;
            obj.ValidationData = validationData;
            obj.ValidationLabels = validationLabels;
            obj.NumGenerations = 0;
            
            %Initialize population
            population = cell(popSize,1);
            inDim = obj.InDim;
            outDim = obj.OutDim;
            
            parfor n = 1:popSize
                population{n} = Candidate(inDim,outDim,0,0,0); %The zeros tell the constructor that the candidate should be initialized randomly
            end
            
            obj.Population = population;
            
        end
        
        %Evolves the population for a fixed number of generations
        function evolve(obj, numGenerations)
            
            %Show progress bar
            wb = waitbar(0,'Evolving:');
            
            for n = 1:numGenerations
                
                %Run through one generation
                obj.computeOneGeneration();
                
                %Print best results so far
                clc;
                disp(['Generation: ', num2str(n)]);
                disp(['Training Fitness: ', num2str(obj.BestTrainingFitness)]);
                disp(['Validation Fitness: ', num2str(obj.BestValidationFitness)]);
                
                %Update progress bar
                waitbar(n/numGenerations,wb);
            end
            
            delete(wb);
            
        end
        
        %Evolve the population through one generation
        function computeOneGeneration(obj)
            
            %Increment generation counter
            obj.NumGenerations = obj.NumGenerations + 1;
            
            %Compute fitness values for each candidate in the population
            fitness = zeros(length(obj.Population),2);
            
            population = obj.Population;
            trainingData = obj.TrainingData;
            trainingLabels = obj.TrainingLabels;
            validationData = obj.ValidationData;
            validationLabels = obj.ValidationLabels;
            
            parfor popInd = 1:length(obj.Population)
                [trainFit, valFit] = population{popInd}.evaluateFitness(trainingData, trainingLabels, validationData, validationLabels);
                fitness(popInd,:) = [trainFit,valFit];
            end
            
            %If better candidate found, or no candidate currently save,
            %than save the best one so far
            [bestPopFitness, index] = max(fitness(:,1));
            
            if isempty(obj.BestCandidate)
                obj.BestTrainingFitness = bestPopFitness;
                obj.BestValidationFitness = fitness(index,2);
                obj.BestCandidate = obj.Population{index};
            end
            
            if (bestPopFitness > obj.BestTrainingFitness)
                obj.BestTrainingFitness = bestPopFitness;
                obj.BestValidationFitness = fitness(index,2);
                obj.BestCandidate = obj.Population{index};
            end
            
            %Fill new population with 1/5 random samples to prevent
            %premature convergence
            newPopulation = cell(obj.PopSize,1);
            inDim = obj.InDim;
            outDim = obj.OutDim;
            
            parfor newPopInd = 1:floor(obj.PopSize / 5)  
                newPopulation{newPopInd} = Candidate(inDim,outDim,0,0,0); 
            end
            
            numCandAdded = floor(obj.PopSize / 5); %The number of candidates added so far to the new population
            
            %Add the best fifth from the current population and the best
            %candidate so far to allow local search around them (elitism)
            [~,bestHalfIndices] = sort(fitness(:,1), 'descend');
            bestHalfIndices = bestHalfIndices(1:floor(end/5));
            
            numToAdd = floor(obj.PopSize / 5);
            start_i = numCandAdded+1;
            end_i = numCandAdded+numToAdd;
            newPopulation(start_i:end_i) = obj.Population(bestHalfIndices(1:(end_i-start_i)+1));
            
            if ~isempty(obj.BestCandidate)
                newPopulation{numCandAdded+numToAdd+1} = obj.BestCandidate;
            end
            
            numCandAdded = numCandAdded + numToAdd + 1; %Update number of candidates added so far
            
            %Fill up population using generalized crossover applied to the
            %best half of the population
            oldPopulation = obj.Population;
            
            parfor n = numCandAdded:obj.PopSize
                %Select two indices from the best half to crossover
                ind1 = randsample(length(bestHalfIndices),1);
                ind2 = randsample(length(bestHalfIndices)-1,1);
                if ind2 == ind1
                    ind2 = ind2 + 1;
                end
                
                %Apply crossover to produce new candidate
                newCandidate = oldPopulation{bestHalfIndices(ind1)}.crossover(oldPopulation{bestHalfIndices(ind2)});
                
                %Add new candidate to new population
                newPopulation{n} = newCandidate;
            end
            
            %Mutate the new population
            parfor candInd = 1:length(newPopulation)   
                newPopulation{candInd}.mutate();
            end
            
            %Replace the old population
            obj.Population = newPopulation;
            
        end
        
        
    end
    
end











