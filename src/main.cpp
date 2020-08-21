#include <iostream>
#include <cstdlib>
#include <opencv4/opencv2/opencv.hpp>
#include <cmath>
#include <ctime>
#include <mpi.h>
#include <string>

using namespace cv;
using namespace std;

// Funcion participantes
void participantes();

// Funcion validadora del formato de imagen
bool isValid(string pathImg);

// Funcion para retornar string con formato de fecha solicitado YYMMDDHHMMSS
string fechaNombre(int opcion);

// Para que numeros simples queden en formato NN ejemplo 8 en 08 utilizada en funcion que genera fecha
string formatoNN(int numero);

// Se cargan variables necesarias para MPI comp globales
int rank;
int size;

// Variables globales para utilizar en envio de datos y recivo de datos a traves de procesadores
const int MAXBYTES=8*1920*1920;
uchar buffer[MAXBYTES];

// FUNCIONES PARA TRABAJAR MAT CON MPI, ENVIO Y RECIVO
// Recordemos que el recivir deja el procesador esperando hasta que el seleccionado haga el envio
void matsnd(Mat& m,int dest);
Mat matrcv(int source);

// Funcion para juntar las imagenes trabajadas (Se divide original en cantidad de procesadores - el orquestador)
void mergeImage(Mat m, Mat & final);

int main(int argc, char** argv )
{       
    // Se recibe el path de la imagen por consola
    String pathIMG = argv[2]; 
    
    // Se cargan variables necesarias para MPI
    int rank;
    int size;      

    // Inicializo ventanas donde se mostraran las imagenes 
    cv::String window_original = "original";
    cv::String window_gray = "gray";
    cv::String window_gauss = "gauss";

    // Inicializo img con la imagen de la ruta
    cv::Mat img = imread(pathIMG);

    if(!isValid(pathIMG)) {
        cout << "Formato no valido" << endl;
        return -1;
    }

    if(img.empty()){
        cout << "Debe ingresar una imagen valida" << endl;
        return -1;
    }

    // Se inicializan objs de la clase Mat que se trabajaran posteriormente
    cv::Mat grayMat;
    cv::Mat gaussImg;
    cv::Mat imgEscalada;

    // Se lee la opcion entregada por consola y se pasa a entero
    int option = (int)*argv[1] - 48;    

    //Caso 1 escala de grises
    //Caso 2 blurr
    //Caso 3 reescalado

    // Si hay menos parametros de los encesarios o mas se termina con error
    if( argc < 3 || argc > 3 ) {
        cout << "Necesita ingresar los parametros necesarios" << endl;
        return -1;
    }

    // Si la opcion no es valida se termina igualmente la ejecucion con error
    if (option<1 || option>3) {
        cout << "Debe ingresar una opcion valida" << endl;
        return -1;
    }
    
    // Inicio de MPI
    MPI_Init(&argc, &argv);

    // Obtencion de rango y numero de procesos del comunicador global (grupo de todos los procesadores de todas las maquinas)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
        
    // Comprueba que haya mas de un procesador, en caso de que sea uno se termina con error
    if(size == 1){
        cout << "Se necesita mas de un procesador" << endl;
        MPI_Finalize(); 
        return -1;
    }

    // Divido la imagen en base a la cantidad de procesadores disponibles para que cada uno se encargue de una parte en paralelo
    // y la envien al orquestador para que las vaya juntando en orden que las va recibiendo.
    int largoRecorte = img.cols/(size-1); 
    
    if(option == 1){
        if (rank == 0) {
            // Ce inicializa imagen final
            Mat finalImg;
            // Contador que utilizaremos para ir recibiendo imagenes en orden
            int NPROC = 1;
            // Por cada "Parte de la imagen" que necesitamos para unirlas corre el siguiente for
            for (int c=0;c < img.cols; c+= largoRecorte) {
                // inicializacion de la porcion de imagen a trabajar
                Mat imgRecortada = img(Rect(c, 0, largoRecorte, img.rows)).clone();
                // se le envia esta porcion al procesador correspondiente
                matsnd(imgRecortada, NPROC);                
                // Pasamos al siguiente procesador              
                NPROC++;
                if(NPROC == size){
                    break;
                }                                                
            }   
            // Con este ciclo for vamos recibiendo todas las imagenes procesadas en orden
            for(int i = 1; i < size;i++) {
                // recibimos la imagen del procesador i
                Mat m = matrcv(i);
                // Se van pegando las imagenes en orden
                mergeImage(m, finalImg);
            }
            string fecha = fechaNombre(option);
            imwrite("../"+ fecha + ".png", finalImg);
            participantes(); 
        }
        // Si no soy el orquestador entro aqui y trabajo lo enviado por el procesador 0
        // En este else todos van a recibir la imagen independiente si el anterio termino
        // por ende trabajaran de forma paralela entre ellos
        else {
            // Recibe del orquestador
            Mat cuttedImg = matrcv(0);            
            Mat finishedImg;
            // Aplica lo solicitado en esta operacion
            cvtColor(cuttedImg,finishedImg,COLOR_BGR2GRAY);
            // Envia lo solicitado al terminar
            matsnd(finishedImg, 0);
        }        
        // Termino de trabajar 
    }

    // Para las otras opciones aplica la misma logica, solo varia la operacion que le aplican
    // a la porcion que se le entrega a cada procesador para trabajar

    if(option == 2){
        if (rank == 0) {
            Mat finalImg;
            int NPROC = 1;
            for (int c=0;c < img.cols; c+= largoRecorte) {
                Mat imgRecortada = img(Rect(c, 0, largoRecorte, img.rows)).clone();
                matsnd(imgRecortada, NPROC);           
                NPROC++;                
                if(NPROC == size){
                    break;
                }                                 
            }    
            for(int i = 1; i < size;i++) {    
                Mat m = matrcv(i);
                mergeImage(m, finalImg);
            }
            string fecha = fechaNombre(option);
            imwrite("../"+ fecha + ".png", finalImg);
            participantes(); 
        }
        else {        
            Mat cuttedImg = matrcv(0);            
            Mat finishedImg;
            GaussianBlur(cuttedImg,finishedImg,Size (9,9),0);                       
            matsnd(finishedImg, 0);
        }    
    }
    if(option == 3){        
        if (rank == 0) {                        
            Mat finalImg;
            int NPROC = 1;
            for (int c=0;c < img.cols; c+= largoRecorte) {
                Mat imgRecortada = img(Rect(c, 0, largoRecorte, img.rows)).clone();
                matsnd(imgRecortada, NPROC);                
                NPROC++;
                if(NPROC == size){                                        
                    break;
                }                            
            }  
            for(int i = 1; i < size;i++) {
                Mat m = matrcv(i);
                mergeImage(m, finalImg);
            }  
            string fecha = fechaNombre(option);
            imwrite("../"+ fecha + ".png", finalImg);
            participantes(); 
        }
        else {        
            Mat cuttedImg = matrcv(0);
            Mat finishedImg;
            Size tamanio(cuttedImg.cols * 2, cuttedImg.rows * 2);
            resize(cuttedImg,finishedImg,tamanio);
            matsnd(finishedImg, 0);
        }
    }

    MPI_Finalize();        
    return 0;
}

void matsnd(Mat& m,int dest){
      int rows  = m.rows;
      int cols  = m.cols;
      int type  = m.type();
      int channels = m.channels();
      memcpy(&buffer[0 * sizeof(int)],(uchar*)&rows,sizeof(int));
      memcpy(&buffer[1 * sizeof(int)],(uchar*)&cols,sizeof(int));
      memcpy(&buffer[2 * sizeof(int)],(uchar*)&type,sizeof(int)); 
      int bytes=m.rows*m.cols*channels;
      if(!m.isContinuous())
      { 
         m = m.clone();
      }
      memcpy(&buffer[3*sizeof(int)],m.data,bytes);
      MPI_Send(&buffer,bytes+3*sizeof(int),MPI_UNSIGNED_CHAR,dest,0,MPI_COMM_WORLD);
}
Mat matrcv(int src){
      MPI_Status status;
      int count,rows,cols,type;
      MPI_Recv(&buffer,sizeof(buffer),MPI_UNSIGNED_CHAR,src,0,MPI_COMM_WORLD,&status);
      MPI_Get_count(&status,MPI_UNSIGNED_CHAR,&count);
      memcpy((uchar*)&rows,&buffer[0 * sizeof(int)], sizeof(int));
      memcpy((uchar*)&cols,&buffer[1 * sizeof(int)], sizeof(int));
      memcpy((uchar*)&type,&buffer[2 * sizeof(int)], sizeof(int));
      // Rearma matriz formato mat
      Mat received= Mat(rows,cols,type,(uchar*)&buffer[3*sizeof(int)]);
      return received;
}
void mergeImage(Mat m, Mat & final) {
    if (final.empty()) {
        final = m.clone();
    }
    else {
        hconcat(final, m, final);
    }
}

string formatoNN(int numero){
    if(numero/10 < 1){
        string retorno = "0" + to_string(numero);
        return retorno;
    }else{
        return to_string(numero);
    }
}

string fechaNombre(int opcion){
    time_t t = std::time(0);   // get time actual
    tm* now = localtime(&t);        
    return ("operacion_" + to_string(opcion) +"_"+ to_string(now->tm_year + 1900) + formatoNN(now->tm_mon + 1) + formatoNN(now->tm_mday) + formatoNN(now->tm_hour) + formatoNN(now->tm_min) + formatoNN(now->tm_sec));    
}

bool isValid(string pathImg){
  vector<string> result;
  for(size_t p=0, q=0; p!=pathImg.npos; p=q) {
    result.push_back(pathImg.substr(p+(p!=0),(q=pathImg.find('.', p+1))-p-(p!=0)));
  }
  if(result.at(result.size()-1) != "gif") return true;
  else return false;
}

void participantes(){
    cout << "" << endl;
    cout << "==========Alumnos Participantes===============" << endl;
    cout << "\t Yerko Foncea Castro" << endl;
    cout << "\t Lucas Carmona Mardones" << endl;
    cout << "\t Brayan Parra Ruz" << endl;
    cout << "==============================================" << endl;
}
