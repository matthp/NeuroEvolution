%Constructs functioning one hidden layer neural network from list
%representation

classdef NeuralNetwork
    
    properties
        
        Wi; %Input layer to hidden layer weight matrix
        Bi; %Hidden layer bias vector
        
        Wo; %Hidden layer to output weight matrix
        Bo; %Output layer bias vector
        
        InDim; %Network input dimensionality
        OutDim; %Network output dimensionality (hard coded 1)
        HidDim; %Network hidden layer dimensionality
        
    end
    
    methods
        
        %Constructor : input variables described above with corresponding
        %property
        function obj = NeuralNetwork(inDim, hidDim, outDim, wi, bi, wo, bo)
            
            %Load instance variables (and decode lists)
            obj.InDim = inDim;
            obj.HidDim = hidDim;
            obj.OutDim = outDim;
            obj.Wi = decodeListIntoMatrix(wi, inDim, hidDim);
            obj.Bi = decodeListIntoMatrix(bi, 1, hidDim);
            obj.Wo = decodeListIntoMatrix(wo, hidDim, outDim);
            obj.Bo = decodeListIntoMatrix(bo, 1, outDim);
            
        end
        
        %Returns outputs for an entire dataset
        function outputs = outputsForDataset(obj, inputs)
            
            inputs = transpose(inputs);
            
            outputs = (obj.Wi*inputs);
            outputs = tanh(bsxfun(@plus, outputs, obj.Bi));
            
            outputs = obj.Wo*outputs;
            outputs = bsxfun(@plus, outputs, obj.Bo);
            
            outputs = transpose(outputs);
            
        end
        
    end
    
end

