//This class is used to create a new canvas to draw on and get the pixel data from it
class scribbleCanvas{

  //A new canvas is created with the specified properties
  constructor(trainingDataDimensions, canvasSize, pixelSize, 
  bkgColor, lineColor, lineWeight){
    this.dataSize = trainingDataDimensions;
    this.canvasSize = canvasSize;
    createCanvas(canvasSize, canvasSize);
    pixelDensity(pixelSize);
    background(bkgColor);
    stroke(lineColor);
    strokeWeight(lineWeight);
  }

  //The image data is collected from the canvas for the Network to use
  getImageData(){
    let img = get();
    img.resize(this.dataSize, this.dataSize);
    img.filter(INVERT);
    img.loadPixels();
    let pixels = [];
    for(let i = 0; i < img.pixels.length; i++){
      if (i % 4 == 0) {
        pixels.push(img.pixels[i]/255);
      }
    }
    //This should be turned on for testing compress quality: image(img, 0, 0);
    return pixels
  }
}

//This class is used to create the user interface
class appUI{
  constructor(dataSize, canvasSize){
    this.dataSize = dataSize;
    this.canvasSize = canvasSize;
    //A new canvas is created using the scribble canvas class
    this.cnv = new scribbleCanvas(this.dataSize, this.canvasSize, 1, "white", "black", 15);
    //The data is cleaned and made usable and the NN is trained
    this.trainData = dataSet.cleanDataSet(100);
    this.epoch = 0;
  }

  //This function draws the user interface
  static interface() {
    //When the mouse is pressed, a line is drawn from the current position to the previous position
    if (mouseIsPressed) line(mouseX, mouseY, pmouseX, pmouseY);
    //When the "space" key is pressed, the canvas is wiped clean
    if (keyIsPressed && key === " ") background("white");    
  }

  //This function is called when the Neural Network is trained or is asked for a prediction
  async useBrain(){
    //When "b" is pressed, a new brain is created with customized traits
    if (keyIsPressed && key === "b") {
      this.machineBrain = new brain(sq(this.dataSize), 0, 28, 10, "Sigmoid");
    }
    //When "p" is pressed, the canvas data is acquired and pushed into the prediction
    if (keyIsPressed && key === "p") {
      let predict = this.machineBrain.prediction([this.cnv.getImageData()]);
      console.log(predict);
    }

    //When "t" is pressed, the NN is trained with the shuffled training data
    if (keyIsPressed && key === "t") {
      this.epoch = await this.machineBrain.train(await this.trainData, this.epoch, 0.1); 
      console.log(`Epoch ${this.epoch} Complete`);   
    }
  }
}