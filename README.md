# PythonProject
My work was the ANN. I have included the explanation of the ANN below from our coursework. I did not create any of the rest of the code in this workbook and thus will not be including any explanation of those parts. Please note the ANN portion of the notebook is at the very END.

--The Following is an excerpt.--
## ANN
### Feature Engineering
MinMaxScaler Transformation:
This transformation method scales the data to a fixed range, typically between 0 and 1. It ensures that each feature contributes proportionally to the model training process, preventing any particular feature from dominating due to its magnitude.

The normalized features after applying MinMaxScaler exhibited consistent scaling, ensuring uniformity across the dataset without altering the relative relationships among different features. This transformation contributes to stabilizing the learning process within the neural network, aiding convergence and preventing the model from being biased towards certain features with larger value ranges.

Inclusion of All Numerical Columns:
The decision to incorporate all available numerical columns without exclusion was driven by the belief that each of these attributes contributes essential information for the predictive task at hand. In our dataset, every numerical attribute portrays distinct characteristics that potentially influence the prediction of the target variable, such as the genre classification.

By retaining all numerical columns, we aimed to leverage the entire spectrum of available information. This approach aligns with the notion that in a domain like music genre classification, various facets of song metadata, including tempo, energy, duration, and others, hold valuable cues influencing the categorization process. Excluding any of these features might lead to information loss and could potentially compromise the model's ability to capture nuanced patterns and correlations among these attributes.

Thus, the inclusive approach was adopted to ensure the neural network could harness the diverse insights encapsulated within the entire set of numerical features, ultimately striving for a more comprehensive and informed predictive model.  
  
  * Output labels: genres

### Architecture

Input Layer:
Dynamically adjusts to the feature size of the input dataset, ensuring that the model can accommodate various data dimensions.
Hidden Layer 1:
Uses the ReLU activation function, known for its effectiveness in introducing non-linearity, improving the learning of complex functions.
Use dropout to reduce overfitting
enhancing the model's ability to generalize to unseen data.
Output Layer:
Number of output layers equals the number of genre's we are attempting to classify.
Employs soft-max activation,
Produces a probability distribution used for group classification
Optimizer:
Adam optimizer is used to help with the convergence in the training process.
Loss Function:
Categorical cross entropy is used since we dealing with a multi-class classification problem
This measures the difference between the predicted probabilities and actual class labels

### Accuracy and Performance
Best Hyperparameters  | Accuracy
--------------------- | -------------
'optimizer': 'adam'   | 0.4597315490245819
'units': 64           | 
'dropout_rate': 0.2   |

### Hyperparameter Tunning
Hidden Layers :
After extensive experimentation with various configurations, the optimal architecture for the Artificial Neural Network (ANN) was determined. Increasing the depth of the network up to 7 layers did not yield substantial accuracy improvements, resulting in overfitting observed during the training phase. As a solution, we decided to choose a simpler ANN architecture was chosen to enhance generalization and improve performance on unknown data.

Neuron Count:
Regarding neuron count variations within layers, different combinations like 128, 64, 256, and 128 were explored. The selection was based on achieving a balanced network depth without compromising performance.

Optimizer:
Optimizer techniques such as Adam, RMSprop, and SGD were tested. Adam demonstrated superior applicability to our dataset due to its adaptive learning rate and momentum, facilitating quicker convergence and efficient parameter updates compared to the other optimizers.

Loss Funtion:
Several loss functions including 'sparse_categorical_crossentropy', 'categorical_crossentropy', and 'kullback_leibler_divergence' were evaluated. The 'categorical_crossentropy' was finalized, aligning with our multi-class classification task where each sample belongs to a single class. This loss function, coupled with softmax activation in the output layer, optimally handles the multi-class scenario by computing the difference between predicted and actual class distributions.

Epochs and Batch Size:
Further fine-tuning involved varying epochs and batch sizes, ranging from epochs=10 to 100 and batch_size=10 to 100. After iterative testing, the optimal combination of epochs=60 and batch_size=32 was determined, striking a balance between convergence speed and memory efficiency."

### Conclusion
The analysis of the confusion matrix sheds light on the ANN's performance in music genre classification. 
Notably, genres such as pop and rap emerge as the most accurately predicted genres. 
Their prevalence within the dataset corresponds directly to the model's higher accuracy in their classification.
However, a critical limitation becomes apparent when discerning between hip hop and rap genres. 
The inherent relationship between these genres, where rap is a sub-genre of hip-hop, presents a challenge for the model. 
The ANN's tendency to misclassify rap songs as hip-hop, and vice versa, underlines the difficulty in defining the nuanced differences between these closely associated genres.

This difficulty in classification arises due to the intricate nature of the genres and their interrelated characteristics within the dataset. 
The model's performance struggles when distinguishing between these interlinked genres, 
reflecting the complexity of genre classifications within music and their representation in the dataset.
The ANN's lower accuracy in predicting genres , coupled with the challenges posed by closely related genres, accentuates the need for more nuanced feature engineering or potentially incorporating additional data facets to enhance the model's discrimination capabilities within closely associated genres.
