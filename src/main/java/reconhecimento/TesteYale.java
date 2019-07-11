package reconhecimento;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;

import java.io.File;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class TesteYale {

    public static void main(String[] args) {

        int totalAcertos = 0;
        double percentualAcerto = 0;
        double totalConfianca = 0;

//        FaceRecognizer reconhecedor = EigenFaceRecognizer.create();
//        FaceRecognizer reconhecedor = FisherFaceRecognizer.create();
        FaceRecognizer reconhecedor = LBPHFaceRecognizer.create();

//        reconhecedor.read("src/main/resources/classificadores/classificador-eigenfaces-yale.yml");
//        reconhecedor.read("src/main/resources/classificadores/classificador-fisherfaces-yale.yml");
        reconhecedor.read("src/main/resources/classificadores/classificador-lbph-yale.yml");

        File diretorio = new File("src/main/resources/yalefaces/teste");
        File[] arquivos = diretorio.listFiles();

        for (File imagem : arquivos) {
            Mat foto = imread(imagem.getAbsolutePath(), IMREAD_GRAYSCALE);
            int classe = Integer.parseInt(imagem.getName().substring(7, 9));
            resize(foto, foto, new Size(160, 160));

            IntPointer rotulo = new IntPointer(1);
            DoublePointer confianca = new DoublePointer(1);
            reconhecedor.predict(foto, rotulo, confianca);
            int predicao = rotulo.get(0);
            System.out.println(classe + " foi reconhecido como " + predicao + " - " + confianca.get(0));
            if (classe == predicao) {
                totalAcertos++;
                totalConfianca += confianca.get(0);
            }
        }

        percentualAcerto = (totalAcertos / 30.0) * 100;
        totalConfianca = totalConfianca / totalAcertos;
        System.out.println("Percentual de acerto: " + percentualAcerto);
        System.out.println("Total confiança: " + totalConfianca);
    }
}
