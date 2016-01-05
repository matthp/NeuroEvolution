%Decodes a list of real values, indexed by integer coordinates into a
%sparse matrix. All values in 'list' take on their real value. All other
%values are zero.

%% Inputs

%'list' is a 2 dimensional array
% - The first column gives the row index in the weight matrix
% - The second column gives the column index in the weight matrix
% - The third column gives the real numbered value the weight takes
% - If the number of columns is only two than the row index becomes the
% column index, and the second column becomes the value.

%'inDim' is the number of rows in the weight matrix

%'outDim' is the number of columns in the weight matrix

%% Outputs

%'weights' is the constructed sparse weight matrix
% - Is two dimensional if 'list' has 3 columns
% - Is one dimensional column vector if 'list' has 2 columns

%% Implementation

function  weights = decodeListIntoMatrix(list, inDim, outDim)
                
    weights = zeros(outDim,inDim);
    
    %Decide if 'list' encodes a weight matrix or a bias vector
    if size(list,2) == 3
        for n = 1:size(list,1)

            x = list(n,1);
            y = list(n,2);

            weights(x,y) = list(n,3);
        end
    else
        for n = 1:size(list,1)
            x = list(n,1);

            weights(x,1) = list(n,2);
        end
    end
    
end

