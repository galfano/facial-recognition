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


import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import java.awt.event.KeyEvent;
import java.util.Scanner;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Created galfano on 05/07/19.
 */
public class Captura {

    public static void main(String args[]) throws FrameGrabber.Exception, InterruptedException {

        KeyEvent tecla = null;

        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();

        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);

        camera.start();

        CascadeClassifier detectorFace = new CascadeClassifier("src/main/resources/haarcascade-frontalface-alt.xml");

        CanvasFrame canvasFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());

        Frame frameCapturado = null;

        Mat imagemColorida = new Mat();

        int numeroAmostras = 25;
        int amostra = 1;

        System.out.println("Digite seu id: ");

        Scanner cadastro = new Scanner(System.in);

        int idPessoa = cadastro.nextInt();

        while ((frameCapturado = camera.grab()) != null) {

            imagemColorida = converteMat.convert(frameCapturado);

            Mat imagemCinza = new Mat();
            cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);

            RectVector facesDetectadas = new RectVector();

            detectorFace.detectMultiScale(imagemCinza,
                    facesDetectadas,
                    1.1,
                    1,
                    0,
                    new Size(150, 150),
                    new Size(500, 500));

            if (tecla == null) {

                tecla = canvasFrame.waitKey(5);
            }

            for (int i = 0; i < facesDetectadas.size(); i++) {
                Rect dadosFace = facesDetectadas.get(0);

                rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0));

                Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                resize(faceCapturada, faceCapturada, new Size(160,160));

                if (tecla == null) {

                    tecla = canvasFrame.waitKey(5);
                }

                if(tecla != null) {

                    if (tecla.getKeyChar() == 'q') {

                        if (amostra <= numeroAmostras) {

                            imwrite("src/main/resources/fotos/pessoa."+idPessoa+"."+amostra+".jpg", faceCapturada);

                            System.out.println("Foto " + amostra + "capturada\n");

                            amostra++;
                        }
                    }

                    tecla = null;
                }
            }

            if (tecla == null) {

                tecla = canvasFrame.waitKey(20);
            }

            if (canvasFrame.isVisible()) {
                canvasFrame.showImage(frameCapturado);
            }

            if(amostra > numeroAmostras) {
                break;
            }
        }

        canvasFrame.dispose();
        camera.stop();
    }
}
