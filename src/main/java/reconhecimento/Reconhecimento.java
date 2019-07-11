/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package reconhecimento;


import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Created galfano on 05/07/19.
 */
public class Reconhecimento {

    public static void main(String args[]) throws FrameGrabber.Exception, InterruptedException {

        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();

        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);

        String[] pessoas = {"", "Collor", "Itamar", "FHC", "Lula", "Dilma", "Temer", "Bolsonaro", "Samuel", "Andre"};

        camera.start();

        CascadeClassifier detectorFace = new CascadeClassifier("src/main/resources/haarcascade-frontalface-alt.xml");

//        FaceRecognizer reconhecedor = EigenFaceRecognizer.create();
//        reconhecedor.read("src/main/resources/classificadores/classificador-eigenfaces.yml");
//        reconhecedor.setThreshold(4000);

//        FaceRecognizer reconhecedor = FisherFaceRecognizer.create();
//        reconhecedor.read("src/main/resources/classificadores/classificador-fisherfaces.yml");
//        reconhecedor.setThreshold(1000);

        FaceRecognizer reconhecedor = LBPHFaceRecognizer.create();
        reconhecedor.read("src/main/resources/classificadores/classificador-lbph.yml");
        reconhecedor.setThreshold(90);


        CanvasFrame canvasFrame = new CanvasFrame("Reconhecimento", CanvasFrame.getDefaultGamma() / camera.getGamma());

        Frame frameCapturado = null;

        Mat imagemColorida = new Mat();

        while ((frameCapturado = camera.grab()) != null) {

            imagemColorida = converteMat.convert(frameCapturado);

            Mat imagemCinza = new Mat();
            cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);

            RectVector facesDetectadas = new RectVector();

            detectorFace.detectMultiScale(imagemCinza,
                    facesDetectadas,
                    1.1,
                    2,
                    0,
                    new Size(150, 150),
                    new Size(500, 500));

            for (int i = 0; i < facesDetectadas.size(); i++) {

                Rect dadosFace = facesDetectadas.get(i);

                rectangle(imagemColorida, dadosFace, new Scalar(0, 255, 0, 0));

                Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                resize(faceCapturada, faceCapturada, new Size(160,160));

                IntPointer rotulo = new IntPointer(1);
                DoublePointer confianca = new DoublePointer(1);

                reconhecedor.predict(faceCapturada, rotulo, confianca);

                int predicao = rotulo.get(0);

                String nome;

                if(predicao == -1) {

                    nome = "Desconhecido";

                } else {

                    nome = pessoas[predicao] + " - " + confianca.get(0);
                }

                int x = Math.max(dadosFace.tl().x() - 10, 0);
                int y = Math.max(dadosFace.tl().y() - 10, 0);
                putText(imagemColorida, nome, new Point(x ,y), FONT_HERSHEY_PLAIN, 1.4, new Scalar(0,255,0,0));

            }

            if (canvasFrame.isVisible()) {
                canvasFrame.showImage(frameCapturado);
            }

        }

        canvasFrame.dispose();
        camera.stop();
    }
}
