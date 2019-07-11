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
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import java.io.File;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Created galfano on 05/07/19.
 */
public class ReconhecimentoSemWebcam {

    public static void main(String args[]) throws FrameGrabber.Exception, InterruptedException {

        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();

        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);

        String[] pessoas = {"", "Joana", "Maria", "Ronaldo", "Wolverine"};

        CascadeClassifier detectorFace = new CascadeClassifier("src/main/resources/haarcascade-frontalface-alt.xml");

//        FaceRecognizer reconhecedor = EigenFaceRecognizer.create();
//        reconhecedor.read("src/main/resources/classificadores/classificador-eigenfaces.yml");
//        reconhecedor.setThreshold(4000);

//        FaceRecognizer reconhecedor = FisherFaceRecognizer.create();
//        reconhecedor.read("src/main/resources/classificadores/classificador-fisherfaces.yml");
//        reconhecedor.setThreshold(1000);

        FaceRecognizer reconhecedor = LBPHFaceRecognizer.create();
        reconhecedor.read("src/main/resources/classificadores/classificador-lbph.yml");
        reconhecedor.setThreshold(100);

        File diretorio = new File("src/main/resources/webcam");
        File[] arquivos = diretorio.listFiles();

        for (File imagem : arquivos) {

            Mat foto = imread(imagem.getAbsolutePath(), IMREAD_GRAYSCALE);

            int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);

            resize(foto, foto, new Size(160, 160));

            IntPointer rotulo = new IntPointer(1);
            DoublePointer confianca = new DoublePointer(1);

            reconhecedor.predict(foto, rotulo, confianca);

            int predicao = rotulo.get(0);

            System.out.println(classe + " foi reconhecido como " + predicao + " - " + confianca.get(0));

        }

    }

}
