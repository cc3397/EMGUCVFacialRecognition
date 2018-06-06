using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using System.Windows;


using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Structure;


namespace emgutest
{
    public enum Emotions{neutral, surprise, sadness, fear, anger, disgust, joy} //define emotion list


    public static class GlobalVars //define some "globals" for use in different functions
    {
        public static int count = 0;
        public static string path = System.IO.Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]); //get debug folder path
        //load all Haar-Cascade object detectors that have been trained to detect faces
        public static CascadeClassifier faceDetectOne = new CascadeClassifier(path + @"\HarrCascade\haarcascade_frontalface_default.xml");
        public static CascadeClassifier faceDetectTwo = new CascadeClassifier(path + @"\HarrCascade\haarcascade_frontalface_alt2.xml");
        public static CascadeClassifier faceDetectThree = new CascadeClassifier(path + @"\HarrCascade\haarcascade_frontalface_alt.xml");
        public static CascadeClassifier faceDetectFour = new CascadeClassifier(path + @"\HarrCascade\haarcascade_frontalface_alt_tree.xml");
        public static FisherFaceRecognizer fishFace = new FisherFaceRecognizer();
        public static List<Emotions> emotion = new List<Emotions>();

        //declare array for loading files /emotions into
        public static int[] emotionForClassifier = new int[608];
        public static Image<Gray, Byte>[] fishFaceImages = new Image<Gray, Byte>[608];

        //declare arrays for training
        public static int predictionInt = (int)(608 * 0.85);

        public static int[] emotionForClassifier_training = new int[predictionInt];
        public static Image<Gray, Byte>[] fishFaceImages_training = new Image<Gray, Byte>[predictionInt];

        //declare array for prediction
        public static int[] emotionForClassifier_prediction = new int[608 - predictionInt];
        public static Image<Gray, Byte>[] fishFaceImages_prediction = new Image<Gray, Byte>[608 - predictionInt];

    }

    class Program
    {

        static void Main(string[] args)
        {


            //define where the emotions are stored
            string[] emotionFolderPath = new string[7];
            emotionFolderPath[0] = @"\Cohn-Kanade Images\000 neutral";
            emotionFolderPath[1] = @"\Cohn-Kanade Images\001 surprise";
            emotionFolderPath[2] = @"\Cohn-Kanade Images\002 sadness";
            emotionFolderPath[3] = @"\Cohn-Kanade Images\003 fear";
            emotionFolderPath[4] = @"\Cohn-Kanade Images\004 anger";
            emotionFolderPath[5] = @"\Cohn-Kanade Images\005 disgust";
            emotionFolderPath[6] = @"\Cohn-Kanade Images\006 joy";

            for (int i = 0; i <= 6; i++)
            {
                LoadImages(GlobalVars.path, emotionFolderPath[i], (Emotions)i); //load each folder of emotion, passing in the corresponding label
            }

            MakeSets(GlobalVars.path); //make the sets for classification

            for (int i = 0; i <= 25; i++) //run 25 times, can be reduced or increased
            {
                RunRecognizer();
            }


        }

        static void LoadImages(string path, string emotionFolderPath, Emotions fileEmotion)
        {

            string[] filePaths = Directory.GetFiles(path + emotionFolderPath); //get path of all images in the folder
            string writePath = path + @"\Cohn-Kanade Images\CutImages\";

            foreach (string imagePath in filePaths)
            {
                Mat img = CvInvoke.Imread(imagePath); //open image
                Mat grayScaleImg = new Mat();
                CvInvoke.CvtColor(img, grayScaleImg, Emgu.CV.CvEnum.ColorConversion.Rgb2Gray); //convert image to greyscale

                //start all Haar-cascade object detectors to look for face in loaded image
                Rectangle[] face = GlobalVars.faceDetectOne.DetectMultiScale(grayScaleImg, 1.1, 10, System.Drawing.Size.Empty, System.Drawing.Size.Empty);
                Rectangle[] faceTwo = GlobalVars.faceDetectTwo.DetectMultiScale(grayScaleImg, 1.1, 10, System.Drawing.Size.Empty, System.Drawing.Size.Empty);
                Rectangle[] faceThree = GlobalVars.faceDetectThree.DetectMultiScale(grayScaleImg, 1.1, 10, System.Drawing.Size.Empty, System.Drawing.Size.Empty);
                Rectangle[] faceFour = GlobalVars.faceDetectFour.DetectMultiScale(grayScaleImg, 1.1, 10, System.Drawing.Size.Empty, System.Drawing.Size.Empty);
                Rectangle[] faceFeatures = null; //initialize to null, will be overwritten by haar-cascade
                //check for the faces being returned
                if (face.Length == 1)
                {
                    faceFeatures = face;
                    Console.WriteLine("Face found in 1 at:" + imagePath);
                }
                else if (faceTwo.Length == 1)
                {
                    faceFeatures = faceTwo;
                    Console.WriteLine("Face found in 2");
                }
                else if (faceThree.Length == 1)
                {
                    faceFeatures = faceThree;
                    Console.WriteLine("Face found in 3");
                }
                else if (faceFour.Length == 1)
                {
                    faceFeatures = faceFour;
                    Console.WriteLine("Face found in 4");
                }
                else
                {
                    Console.WriteLine("No Face Found");
                }

                try //try to output image and emotion label, if it fails, catch the error and continue processing
                {
                    Image<Gray, Byte> outputImg = grayScaleImg.ToImage<Gray, Byte>();
                    outputImg.ROI = faceFeatures[0];
                    System.Drawing.Size imgSize = new System.Drawing.Size();
                    imgSize.Height = 200;
                    imgSize.Width = 200;
                    CvInvoke.Resize(outputImg, outputImg, imgSize);
                    CvInvoke.Imwrite(writePath + GlobalVars.count + ".png", outputImg);
                    GlobalVars.emotion.Add(fileEmotion); //add emotion to emotions database
                    Console.WriteLine("Emotion Loaded As: " + GlobalVars.emotion[GlobalVars.count]);

                }
                catch
                {
                    Console.WriteLine("Error in writing image to disk");
                    //return;
                }
                GlobalVars.count++;
            }

        }

        static void MakeSets(string rootPath) //load the rescaled images from the disk and put them in appropriate array
        {
            string[] filePaths = Directory.GetFiles(rootPath + @"\Cohn-Kanade Images\CutImages\"); //retrive cropped images
            int i = 0;
            foreach (string imagePath in filePaths)
            {
                Mat loadedImage = CvInvoke.Imread(rootPath + @"\Cohn-Kanade Images\CutImages\" + i + ".png");
                Image<Gray, Byte> img = loadedImage.ToImage<Gray, Byte>();
                GlobalVars.fishFaceImages[i] = img;
                GlobalVars.emotionForClassifier[i] = (int) (GlobalVars.emotion[i]);
                i++;
            }
            
        }

        static void RandomizeSets()//randomize the array
        {
            Random rnd1 = new Random();
            
            for (int x = 0; x < 608; x++)
            {
                //rnd values for use in loop to ensure the emotion and related image stay in same place in arrays
                int looprnd = rnd1.Next(0, 608);
                //temp values to store overwitten image
                int temp = GlobalVars.emotionForClassifier[x];
                Image<Gray, Byte> tempImg = GlobalVars.fishFaceImages[x];
                //shuffle emotion
                GlobalVars.emotionForClassifier[x] = GlobalVars.emotionForClassifier[looprnd];
                GlobalVars.emotionForClassifier[looprnd] = temp;
                //shuffle images
                GlobalVars.fishFaceImages[x] = GlobalVars.fishFaceImages[looprnd];
                GlobalVars.fishFaceImages[looprnd] = tempImg;

            }
            
            //now split set into training images and classification images (85% for training, 15% for prediction)
            for (int x = 0; x < (int)(GlobalVars.emotionForClassifier.Length * 0.85); x++)
            {
                GlobalVars.emotionForClassifier_training[x] = GlobalVars.emotionForClassifier[x];
                GlobalVars.fishFaceImages_training[x] = GlobalVars.fishFaceImages[x];
            }

            for (int x = (int)(GlobalVars.emotionForClassifier.Length * 0.85); x < GlobalVars.emotionForClassifier.Length; x++)
            {
                GlobalVars.emotionForClassifier_prediction[x - ((int)(GlobalVars.emotionForClassifier.Length * 0.85))] = GlobalVars.emotionForClassifier[x];
                GlobalVars.fishFaceImages_prediction[x - ((int)(GlobalVars.fishFaceImages.Length * 0.85))] = GlobalVars.fishFaceImages[x];
            }
            
        }


        static void RunRecognizer()
        {
            RandomizeSets(); //re - randomize the set when training again
           
            //train the set
            Console.WriteLine("Training set");
            GlobalVars.fishFace.Train(GlobalVars.fishFaceImages_training, GlobalVars.emotionForClassifier_training);

            int count = 0;
            int amountCorrect = 0;
            foreach (Image<Gray, Byte> img in GlobalVars.fishFaceImages_prediction) //predict all images in the prediction set
            {
                Emgu.CV.Face.FaceRecognizer.PredictionResult prediction = GlobalVars.fishFace.Predict(img);
                
               // Console.WriteLine("Predicted emotion is: " + (Emotions)prediction.Label + " With a " + prediction.Distance + " Distance. Actual emotion is " + (Emotions)GlobalVars.emotionForClassifier_prediction[count]);
                if(GlobalVars.emotionForClassifier_prediction[count] == prediction.Label) //check to see if the predicted emotion is correct
                {
                    amountCorrect++;
                }
                else //write the wrong image to disk for inspection, can comment out if this feature is not required
                {
                    CvInvoke.Imwrite(GlobalVars.path + @"\Cohn-Kanade Images\IncorrectImages\" + (Emotions)prediction.Label + (Emotions)GlobalVars.emotionForClassifier_prediction[count] + count + ".png", img);
                }
                count++;
            }
           
            Console.WriteLine("Got " + amountCorrect + " out of " + count + " images"); //inform user the amount that was correctly predicted
           
        }

   }
}


