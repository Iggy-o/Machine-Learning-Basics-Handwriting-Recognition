class layer{
  // When this class is initialized a set of weights and biases are created
  constructor (inputs, neurons, activationType){

    //Two empty arrays for the weights and biases are created 
    this.weights = [];
    this.biases = [];
    this.activationType = activationType;

    //They are initially filled with random numbers which will be optimized by the algorithm
    for (let i = 0; i < neurons; i++) {
      //A new bias is created for every specified neuron
      this.biases.push((Math.random() * 2) - 1);
      
      //A new set of weights is created for every neuron, one for each input
      this.weights.push([])
      for (let j = 0; j < inputs; j++) {
        this.weights[i].push((Math.random() * 2) - 1);
      }
    }
  }

  // The layer's output is calculated and returned
  output(inputs){
    // The output variable is initialized and the neuron values are pushed into it via the loop
    let layerOutputs = [];
    for (let i = 0; i < inputs.length; i++){
      layerOutputs.push([])
      for (let j = 0; j < this.weights.length; j++) {
        //The neuron value is equal to "dot"
        //Dot is a function which multiplies the inputs by their associated weights and adds them all together to get the value of one specific neuron
        let neuronVal = math.dot(this.weights[j], inputs[i]) + this.biases[j];
        // The neuron's output is run through the relu activation function before being added to the layer output array
        let squashedOutput = brain.activation(neuronVal, this.activationType);
        layerOutputs[i].push(squashedOutput);
      }
    }
    return layerOutputs
  }

  //Using the training data the neural networks error is reduced
  errorSupress(error, sigmoidDerivative, learningRate){
    //Add previouslayer transposed/Add proper list of sigmoid derivative

    //The weights and biases are adjusted to shrink the error 
    //and the hidden error is calculated and returned for the next layer to use
    let hiddenError = [];
    hiddenError.length = this.weights[0].length;
    hiddenError.fill(0);
    for (let i = 0; i < this.biases.length; i++) {
      hiddenError = math.add(hiddenError, math.multiply(this.weights[i], error[i]));
      for (let j = 0; j < this.weights[i].length; j++) {
        this.weights[i][j] -= error[i]*sigmoidDerivative[i]*learningRate; //*previousLayer
      }
      this.biases[i] -= error[i]*sigmoidDerivative[i]*learningRate;
    }
    return hiddenError
  }
}

class brain{

  //When a new brain is initialized it must recieve costumizable traits
  constructor(startingInputs, hiddenlayers, hiddenlayer_neurons, endlayer_outputs, activationType){
    // The neural networks layers are constructed using the layers class and put in the layers array
    this.layers = [];
    //The input layer is created
    this.layers.push(
      new layer(startingInputs, hiddenlayer_neurons, activationType)
    );
    //The indicated number of hidden layers are created
    for (let i = 0; i < hiddenlayers; i++) {
      this.layers.push(
        new layer(hiddenlayer_neurons, hiddenlayer_neurons, activationType)
      );
    }
    //The output layer is created
    this.layers.push(
      new layer(hiddenlayer_neurons, endlayer_outputs, activationType)
    );
  }

  //The selected activation function is used by the NN
  static activation(value, activationType) {
    switch(activationType) {
      // This is an activation function that returns 0 if the output is less than a certain value
      case "ReLu":
        return Math.max(0, value);
        break;
      //This is a calculation of sigmoids derivative using a value that has already been pushed through sigmoid
      case "dSigmoid":
        return (value) * (1 - value);
        break;
      //This is an activation function that squishes numbers between 0 and 1 but they keep their meaning
      case "Sigmoid":
        return 1/(1+Math.pow(Math.E, -value));
        break;
      //The default activation is set to Sigmoid
      default:
        return 1/(1+Math.pow(Math.E, -value));
    }
  }

  //The data is given to the network for it to make a prediction
  forwardPropagation(data, returnLayerOutputs = false) {
    let layerReturn = this.layers[0].output(data);
    let outputs = []
    for (let i = 1; i < this.layers.length; i++){
      outputs = outputs.concat(layerReturn);
      layerReturn = this.layers[i].output(layerReturn);
    }
    outputs = outputs.concat(layerReturn);
    if (returnLayerOutputs == true) return outputs;
    else return layerReturn;
  }

  //The inputted data is pushed into the forward propagation function and the result is converted into a number
  prediction(data){
    let prediction = this.forwardPropagation(data);
    //console.log(prediction)
    let maxPrediction = -1;
    let predictedNumber;
    for (let i = 0; i < 10; i++){
      if (Math.max(maxPrediction, prediction[0][i]) == prediction[0][i]) {
        maxPrediction = prediction[0][i];
        predictedNumber = i;
      }
    }
    return predictedNumber
  }

  //This function is used to train the NN based on the given dataset
  async train(trainData, epoch, learningRate) {
    for (let i = 0; i < trainData.length; i++) {
      //console.log(trainData)
      //The training data's label is removed from the data and stored for other purposes
      let trainLabel = trainData[i][trainData[i].length - 1];
      trainData[i].pop();
      let data = trainData[i];
      //The prediction is used to calculate the brain's error
      let prediction = this.forwardPropagation([data]);
      let layerOutputs = this.forwardPropagation([data], true);
      let error = [];
      for (let j = 0; j < prediction[0].length; j++) {
        //console.log(error);
        error.push(sq(math.subtract(trainLabel[j], prediction[0][j])));
      }
      //The given error is used to nudge the neural network in the right direction
      await this.backPropagation(error, layerOutputs, learningRate) 
      //console.log("done")
    }
    return ++epoch;
  }

  //This function tries to limit the neural networks error
  async backPropagation(error, layerOutputs, learningRate){
    //The derivative of sigmoid are gathered and used in the error supression
    let dSigmoids = [];
    for (let i = 0; i < layerOutputs.length; i++) {
      dSigmoids.push([])
      for (let j = 0; j < layerOutputs[i].length; j++) {
        //console.log(layerOutputs[i][j])
        dSigmoids[i].push(brain.activation(layerOutputs[i][j], "dSigmoid"));
      }
    }
    //The error is passed in to each layer for them to adjust their weights
    let layerError = await this.layers[this.layers.length - 1].errorSupress(error, dSigmoids[this.layers.length - 1], learningRate);
    for (let i = this.layers.length - 2; i >= 0; i--){
      layerError = this.layers[i].errorSupress(layerError, dSigmoids[i], learningRate);
    }
  }
}