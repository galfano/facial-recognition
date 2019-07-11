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

import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

/**
 * Created galfano on 05/07/19.
 */
public class TreinamentoYale {

    public static void main(String args[]) {

        File diretorio = new File("src/main/resources/yalefaces/treinamento");

        File[] arquivos = diretorio.listFiles();

        MatVector fotos =  new MatVector(arquivos.length);

        Mat rotulos = new Mat(arquivos.length, 1, CV_32SC1);

        IntBuffer rotulosBuffer = rotulos.createBuffer();

        int contador = 0;

        for(File imagem: arquivos) {

            Mat foto = imread(imagem.getAbsolutePath(), IMREAD_GRAYSCALE);

            int classe = Integer.parseInt(imagem.getName().substring(7,9));

            resize(foto, foto, new Size(160,160));

            fotos.put(contador, foto);

            rotulosBuffer.put(contador, classe);

            contador++;
        }

        FaceRecognizer eigenfaces = EigenFaceRecognizer.create(30, 0);
        FaceRecognizer fisherfaces = FisherFaceRecognizer.create(30, 0);
        FaceRecognizer lbph = LBPHFaceRecognizer.create(12, 10, 15, 15, 0);

        eigenfaces.train(fotos, rotulos);
        eigenfaces.save("src/main/resources/classificadores/classificador-eigenfaces-yale.yml");

        fisherfaces.train(fotos, rotulos);
        fisherfaces.save("src/main/resources/classificadores/classificador-fisherfaces-yale.yml");

        lbph.train(fotos, rotulos);
        lbph.save("src/main/resources/classificadores/classificador-lbph-yale.yml");
    }
}
