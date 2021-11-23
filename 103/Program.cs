using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Microsoft.VisualBasic.FileIO;
using MPI;

namespace _103
{
    class Program
    {
        const string file = @"C:\Users\gnede\source\repos\103\103\bin\Debug\netcoreapp3.1\Lab_139_1Ch1.csv";

        static void Main(string[] args)
        {
            using (new MPI.Environment(ref args))
            {
                var comm = Communicator.world;
                double[,] data = null;

                //Отправка данных всем процессам
                if (comm.Rank == 0)
                    data = GetData(file);
                comm.Broadcast(ref data, 0);

                //Выполнение алгоритма K-means
                int n = 5;                                  //Число выполнений алгоритма K-means
                int N = 50;                                //Число кластеров
                int[] sol;                                  //Решение
                double index = double.PositiveInfinity;     //RMSSTD-индекс
                for (int i = 0; i < n; i++)
                {
                    double tmp = K_means(data, N, out int[] tmp1);
                    Console.WriteLine("Ранг процесса:" + Communicator.world.Rank + ", процесс запущен на: " + MPI.Environment.ProcessorName + ", RMSSTD-индекс: " + tmp);
                    if (index > tmp)
                    {
                        sol = tmp1;
                        index = tmp;
                    }
                }

                //Получение минимального значения индекса
                double minIndex = comm.Reduce(index, Operation<double>.Min, 0);
                comm.Broadcast(ref minIndex, 0);
                if (index == minIndex)
                    Console.WriteLine("\nЛучший результат: RMSSTD-индекс = " + index + ", ранг процесса:" + Communicator.world.Rank + ", процесс запущен на: " + MPI.Environment.ProcessorName);
            }
        }
        
        static double[,] Rand(int N, int n)
        {
            double[,] arr = new double[N, n];
            Random rand = new Random();
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    arr[i, j] = rand.NextDouble();
                }
            }
            return arr;
        }

        //Получение массива данных из файла
        static double[,] GetData(string path)
        {
            List<double[]> data = new List<double[]>();
            System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
            using (TextFieldParser parser = new TextFieldParser(path))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(";");
                var culture = CultureInfo.CreateSpecificCulture("en-NZ");
                while (!parser.EndOfData)
                {
                    string[] fields = parser.ReadFields();
                    if (parser.LineNumber > 4)
                    {
                        double[] tmp = new double[fields.Length - 1];
                        for (int i = 0; i < fields.Length - 1; i++)
                        {
                            double n;
                            double.TryParse(fields[i], NumberStyles.Float, culture, out n);
                            tmp[i] = n;
                        }
                        data.Add(tmp);
                    }
                }
            }
            return Rationing(data);
        }

        //Приведение атрибутов к диапозону от 0 до 1
        static double[,] Rationing(List<double[]> arr)
        {
            int x = arr.Count;
            int y = arr[0].Length;
            double[,] res = new double[x, y];
            double[] max = new double[y];
            double[] min = new double[y];

            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    if (arr[i][j] > max[j] || i == 0)
                        max[j] = arr[i][j];
                    if (arr[i][j] < min[j] || i == 0)
                        min[j] = arr[i][j];
                }
            }
            for (int i = 0; i < x; i++)
                for (int j = 0; j < y; j++)
                    res[i, j] = (arr[i][j] - min[j]) / (max[j] - min[j]);
            return res;
        }

        static double K_means(double[,] arr, int n, out int[] clust)
        {
            //Выбор случайных начальных центров кластеров
            Random rand = new Random();
            double[,] c = new double[n, arr.GetLength(1)];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < arr.GetLength(1); j++)
                    c[i, j] = rand.NextDouble();

            double[] dist = new double[arr.GetLength(0)];   //Расстояния до кластеров
            clust = new int[arr.GetLength(0)];              //Кластеры

            while (true)
            {
                //Вычисление расстояний до центров кластеров и перерасчет кластеров
                int s = 0;
                for (int i = 0; i < arr.GetLength(0); i++)
                {
                    int tmp1 = 0;
                    for (int m = 0; m < n; m++)
                    {
                        double tmp = 0;
                        for (int j = 0; j < arr.GetLength(1); j++)
                        {
                            tmp += Math.Pow(arr[i, j] - c[m, j], 2);
                        }
                        if (dist[i] > tmp || m == 0)
                        {
                            tmp1 = m;
                            dist[i] = tmp;
                        }
                    }
                    if (clust[i] == tmp1)
                        s++;
                    clust[i] = tmp1;
                }

                //Если нет изменений, закончить цикл
                if (s == arr.GetLength(0))
                    break;

                //Расчет новых центров кластеров
                c = new double[n, arr.GetLength(1)];
                int[] count = new int[n];
                for (int i = 0; i < arr.GetLength(0); i++)
                {
                    count[clust[i]]++;
                    for (int j = 0; j < arr.GetLength(1); j++)
                        c[clust[i], j] += arr[i, j];
                }
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < arr.GetLength(1); j++)
                        c[i, j] /= count[i];
            }
            return RMSSTD(arr, c, clust);
        }
        static double RMSSTD(double[,] arr, double[,] c, int[] clust)
        {
            double res = 0;
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                double tmp = 0;
                for (int j = 0; j < arr.GetLength(1); j++)
                    tmp += Math.Pow(arr[i, j] - c[clust[i], j], 2);
                res += Math.Sqrt(tmp);
            }
            res /= arr.GetLength(1) * (arr.GetLength(0) - c.GetLength(0));
            return Math.Sqrt(res);
        }
    }
}
