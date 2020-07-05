let canvasSize = 150;
let dataImageSize = 28;

//The test data is preloaded using a p5 function called preload
function preload(){
  //The setup data class is initialized and data is loaded into the program
  dataSet = new setupData(dataImageSize, canvasSize);
  for (let i = 9; i >= 0; i--) dataSet.loadData(`test/mnist/data${i}`);
}

//Setup is a p5 function used to initialize objects like canvases
async function setup(){
  //A new app canvas is created and the dataset if prepped
  //noLoop();
  app = new appUI(dataImageSize, canvasSize);
  //After the data is prepared the last drawing is wiped from the canvas
  await app.trainData;
  background("white");
  //loop();
}

//Draw is also a p5 function used to repeat commands every second and in my case draw to the canvas
function draw(){
    //The UI elements are created
    appUI.interface();
    //The Neural Network is called and used
    app.useBrain();
}


    
