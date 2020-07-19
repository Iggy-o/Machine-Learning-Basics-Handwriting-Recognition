//This is a propietary function tailored to convert handwritten number training labels to usable data
class setupData{
  constructor(imageSize, canvasSize){
    this.uncleanData = []
    this.data = [];
    this.imageSize = imageSize;
    this.canvasSize = canvasSize;
  }

  loadData(file){
    this.uncleanData.push(loadBytes(file))
  }

  async cleanDataSet(amount){
    console.log("preparing data...");
    for (let i = 0; i < this.uncleanData.length; i++){
      let data = this.uncleanData[i];
      let optimizedlabel = [];
      optimizedlabel.length = 10;
      optimizedlabel.fill(0);
      optimizedlabel[i] = 1;
      listPixels = [];
      await this.drawData(data, optimizedlabel, amount);
      this.data = this.data.concat(listPixels);
    }
    console.log("data ready!!!");
    return this.data//shuffle(this.data, true)
  }

  async drawData(data, label, amount){
    let imageNum = 0;
    for (let h = 0; h < data.bytes.length/amount; h+=784){
      await this.sleep(0);
      let dataImg = createImage(28, 28);
      dataImg.loadPixels();
      listPixels.push([]);
      for (let i = 0; i < dataImg.pixels.length; i++){
        let index = i * 4;
        if (i%4 == 0) listPixels[imageNum].push(data.bytes[i + imageNum]/255);
        for (let j = 0; j < 3; j++) {
          dataImg.pixels[index + j] = data.bytes[i + h];
        }
        dataImg.pixels[index + 3] = 255;
      }
      listPixels[imageNum].push(label)
      imageNum++;
      dataImg.updatePixels();
      dataImg.filter(INVERT);
      dataImg.resize(this.canvasSize, 0);
      image(dataImg, 0, 0);
    }
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
let listPixels;