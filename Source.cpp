#include<iostream>
#include<fstream>
#include <string>
#include <time.h>
#include <algorithm>
#include <vector>
#include <ctime>

using namespace std;


char* substr(char* arr, int begin, int len)
{
	char* res = new char[len];
	for (int i = 0; i < len; i++)
		res[i] = *(arr + begin + i);
	res[len] = 0;
	return res;
}

void sleepcp(int milliseconds) // cross-platform sleep function
{
	clock_t time_end;
	time_end = clock() + milliseconds * CLOCKS_PER_SEC / 1000;
	while (clock() < time_end)
	{
	}
}

string convert(string str_in)
{
	for (int i = 0; i < str_in.length(); i++)
	{
		switch (str_in[i])
		{
		case '\\':
			str_in[i] = '\\';
			string str = "\\";
			str_in.insert(i, str);
			i++;
			break;
		}
	}
	return str_in;
}

int main() {
	//emotion for classification
	int emotion = 4;
	int state = 2;
	bool select = true;//if false, uses false_emotion
	int false_emotion = 2;
	if (state == 1){
		//Read paths to label .txt files, convert to paths for -simalign output
		ifstream raw_path, imgdir_path;
		ofstream simalign_batch;

		raw_path.open("C:\\Users\\Dell\\Emotion\\label_path.txt");//also holds feature point coordinates; 1st line = label, 2nd line = neutral emotion(x1 y1, x2, y2 ...), 3rd line = full emotion
		imgdir_path.open("C:\\Users\\Dell\\Emotion\\imgdir_path.txt");
	
	
		char label_path[100];
		string simalign[328];
		string label_loc[328];

		//generate simalign paths
		if (raw_path.is_open()){
			int counter = 0;
			while (!raw_path.eof()){
				string simalign_path;
				raw_path >> label_path;
				label_loc[counter] = label_path;
				simalign_path = substr(label_path, 0, 30);
				//simalign_path = convert(simalign_path);

				simalign[counter] = simalign_path;
				counter++;
			
			}
		}

		char imgdir[100];
		string imgdir_final[328];

		//generate db images paths
		if (imgdir_path.is_open()){
			int counter = 0;
			while (!imgdir_path.eof()){
				string imgdir_temp;
				imgdir_path >> imgdir;
				imgdir_temp = substr(imgdir, 0, 69);
				//imgdir_temp = convert(imgdir_temp);

				imgdir_final[counter] = imgdir_temp;
				counter++;
			}
		}

		//generate batch for simalign
		simalign_batch.open("C:\\Users\\Dell\\Documents\\tere\\code\\FERA-2015\\C++ models\\Release\\simalign_batch.bat");
		string line = "";
		if (simalign_batch.is_open()){
			for (int i = 0; i < 327; i++){//0..326
				line = "FeatureExtraction.exe -fdir \"" + imgdir_final[i] + "\" -simalign \"" + simalign[i] + "\"";
				simalign_batch << line << endl;
			}
		}

		//generate batch for extracting features of simaligned images
		ofstream extract_batch;
		extract_batch.open("C:\\Users\\Dell\\Documents\\tere\\code\\FERA-2015\\C++ models\\Release\\extract_batch.bat");
		if (extract_batch.is_open()){
			for (int i = 0; i < 327; i++){//0..326
				line = "FeatureExtraction.exe -fdir \"" + simalign[i] + "\" -of \"" + label_loc[i] + "\"";
				extract_batch << line << endl;
			}
		}

		//extract data
		int label[328];
		float neutral_x[328][68];
		float neutral_y[328][68];
		float full_x[328][68];
		float full_y[328][68];

		for (int i = 0; i < 327; i++){//0..326
			ifstream feature_file;
			string path_convert;
			path_convert = convert(label_loc[i]);
			feature_file.open(path_convert);

			int n_x = 0;
			int n_y = 0;
			int f_x = 0;
			int f_y = 0;
			if (feature_file.is_open()){
				int counter = 0;
				while (!feature_file.eof()){
					char raw_data[600];
					string data;
					if (counter == 0){//labels extracted
						feature_file >> raw_data;
						data = substr(raw_data, 0, 1);
						label[i] = stoi(data);
					
					}
					else if ((counter > 0)&&(counter < 137)){//neutral emotion range
						feature_file >> raw_data;
						if (counter % 2 == 1){
							neutral_x[i][n_x] = stof(raw_data);
							n_x++;
						}
						else{
							neutral_y[i][n_y] = stof(raw_data);
							n_y++;
						}
					

					}
					else if ((counter >136)&&(counter < 273)){//full emotion range
						feature_file >> raw_data;
						if (counter % 2 == 1){
							full_x[i][f_x] = stof(raw_data);
							f_x++;
						}
						else{
							full_y[i][f_y] = stof(raw_data);
							f_y++;
						}
					}
					else{
						feature_file >> raw_data;
					}

					counter++;
				}
			}
		
		}

		//calculate distances
		float distances[328][68];
		float min_distance[327];
		fill(min_distance, min_distance + 327, 1000);
		float max_distance[327];
		fill(max_distance, max_distance + 327, 0);
		for (int i = 0; i < 327; i++){
			for (int j = 0; j < 68; j++){
				float x0 = neutral_x[i][j];
				float x1 = full_x[i][j];

				float y0 = neutral_y[i][j];
				float y1 = full_y[i][j];
			
				float dist_x = (x1 - x0)*(x1 - x0);
				float dist_y = (y1 - y0)*(y1 - y0);

				float dist = sqrt(dist_x + dist_y);
				distances[i][j] = dist;

				if (dist > max_distance[i]){
					max_distance[i] = dist;
				}
				if (dist < min_distance[i]){
					min_distance[i] = dist;
				}
			}
		}
		/*
		for (int i = 0; i < 327; i++){
			for (int j = 0; j < 68; j++){
				distances[i][j] = (distances[i][j]) / (max_distance[i]);
			}
		}*/
		
		//separate by emotion
		float angry[45][68];
		float contempt[18][68];
		float disgust[59][68];
		float fear[25][68];
		float happy[69][68];
		float sadness[28][68];
		float surprise[83][68];

		int an = 0;
		int co = 0;
		int di = 0;
		int fe = 0;
		int ha = 0;
		int sa = 0;
		int su = 0;
	
		for (int i = 0; i < 327; i++){
			if (label[i] == 1){
				for (int j = 0; j < 68; j++){
					angry[an][j] = distances[i][j];
				}
				an++;
			}
			else if (label[i] == 2){
				for (int j = 0; j < 68; j++){
					contempt[co][j] = distances[i][j];
				}
				co++;
			}
			else if (label[i] == 3){
				for (int j = 0; j < 68; j++){
					disgust[di][j] = distances[i][j];
				}
				di++;
			}
			else if (label[i] == 4){
				for (int j = 0; j < 68; j++){
					fear[fe][j] = distances[i][j];
				}
				fe++;
			}
			else if (label[i] == 5){
				for (int j = 0; j < 68; j++){
					happy[ha][j] = distances[i][j];
				}
				ha++;
			}
			else if (label[i] == 6){
				for (int j = 0; j < 68; j++){
					sadness[sa][j] = distances[i][j];
				}
				sa++;
			}
			else if (label[i] == 7){
				for (int j = 0; j < 68; j++){
					surprise[su][j] = distances[i][j];
				}
				su++;
			}
		}

		//generate files for SVM calssification

		an = 45;
		co = 18;
		di = 59;
		fe = 25;
		ha = 69;
		sa = 28;
		su = 83;

		int iterator;
		int o_iterator = 0;
		float selected_emotion[90][68];
		float other_emotions[330][68];

		
		if (emotion == 1){
			iterator = an;
			for (int i = 0; i<an; i++){
				for (int j = 0; j < 68; j++){
					selected_emotion[i][j] = angry[i][j];
				}
			}
		
			for (int i = 0; i < 327 - iterator; i++){
				if (label[i] == 1){

				}
				else{
					for (int j = 0; j < 68; j++){
						other_emotions[o_iterator][j] = distances[i][j];
					}
					o_iterator++;
				}
			}
		}
		else if (emotion == 2){
			iterator = co;
			for (int i = 0; i<co; i++){
				for (int j = 0; j < 68; j++){
					selected_emotion[i][j] = contempt[i][j];
				}
			}
			for (int i = 0; i < 327 - iterator; i++){
				if (label[i] == 2){

				}
				else{
					for (int j = 0; j < 68; j++){
						other_emotions[o_iterator][j] = distances[i][j];
					}
					o_iterator++;
				}
			}
		}
		else if (emotion == 3){
			iterator = di;
			for (int i = 0; i<di; i++){
				for (int j = 0; j < 68; j++){
					selected_emotion[i][j] = disgust[i][j];
				}
			}
			for (int i = 0; i < 327 - iterator; i++){
				if (label[i] == 3){

				}
				else{
					for (int j = 0; j < 68; j++){
						other_emotions[o_iterator][j] = distances[i][j];
					}
					o_iterator++;
				}
			}
		}
		else if (emotion == 4){
			iterator = fe;
			for (int i = 0; i<fe; i++){
				for (int j = 0; j < 68; j++){
					selected_emotion[i][j] = fear[i][j];
				}
			}
			for (int i = 0; i < 327 - iterator; i++){
				if (label[i] == 4){

				}
				else{
					for (int j = 0; j < 68; j++){
						other_emotions[o_iterator][j] = distances[i][j];
					}
					o_iterator++;
				}
			}
		}
		else if (emotion == 5){
			iterator = ha;
			for (int i = 0; i<ha; i++){
				for (int j = 0; j < 68; j++){
					selected_emotion[i][j] = happy[i][j];
				}
			}
			for (int i = 0; i < 327 - iterator; i++){
				if (label[i] == 5){

				}
				else{
					for (int j = 0; j < 68; j++){
						other_emotions[o_iterator][j] = distances[i][j];
					}
					o_iterator++;
				}
			}
		}
		else if (emotion == 6){
			iterator = sa;
			for (int i = 0; i<sa; i++){
				for (int j = 0; j < 68; j++){
					selected_emotion[i][j] = sadness[i][j];
				}
			}
			for (int i = 0; i < 327 - iterator; i++){
				if (label[i] == 6){

				}
				else{
					for (int j = 0; j < 68; j++){
						other_emotions[o_iterator][j] = distances[i][j];
					}
					o_iterator++;
				}
			}
		}
		else if (emotion == 7){
			iterator = su;
			for (int i = 0; i<su; i++){
				for (int j = 0; j < 68; j++){
					selected_emotion[i][j] = surprise[i][j];
				}
			}
			for (int i = 0; i < 327 - iterator; i++){
				if (label[i] == 7){

				}
				else{
					for (int j = 0; j < 68; j++){
						other_emotions[o_iterator][j] = distances[i][j];
					}
					o_iterator++;
				}
			}
		}
	


		for (int i = 0; i < iterator; i++){
			ofstream test;
			ofstream data;
			test.open("C:\\Users\\Dell\\Downloads\\libsvm-3.20\\libsvm-3.20\\windows\\test\\test" + to_string(i) + ".txt");
			data.open("C:\\Users\\Dell\\Downloads\\libsvm-3.20\\libsvm-3.20\\windows\\data\\data" + to_string(i) + ".txt");
			if (test.is_open()){
				string line = "+1 ";
				for (int j = 0; j < 68; j++){
					//if (selected_emotion[i][j]>0.1){//leia mind
						line.append(to_string(j) + ":" +to_string(selected_emotion[i][j]) + " ");
					//}
				}
				test << line;
			}
			if (data.is_open()){
				for (int y = 0; y < iterator; y++){
					if (y == i){

					}
					else{
						string line = "+1 ";
						for (int j = 0; j < 68; j++){
							line = line + to_string(j) + ":" + to_string(selected_emotion[y][j]) + " ";
						}
						data << line << endl;
					}
				}

				for (int y = 0; y < (327 - o_iterator); y++){
					string line = "-1 ";
					for (int j = 0; j < 68; j++){
						line = line + to_string(j) + ":" + to_string(other_emotions[y][j]) + " ";
					}
					data << line << endl;
				}
			}
		
		}

		ofstream classify_batch;
		classify_batch.open("C:\\Users\\Dell\\Downloads\\libsvm-3.20\\libsvm-3.20\\windows\\classify.bat");
		if (classify_batch.is_open()){
			for (int i = 0; i < iterator; i++){
				string line = "svm-train.exe  data\\data" + to_string(i) + ".txt";
				classify_batch << line << endl;
				line = "svm-predict.exe test\\test" + to_string(i) + ".txt data\\data" + to_string(i) + ".txt.model results\\result" + to_string(i) + ".txt";
				classify_batch << line << endl;
			}
		}
	}
	else if (state = 2){
	
		int iterator = 0;

		if (emotion == 1){
			iterator = 45;
		}
		else if (emotion == 2){
			iterator = 18;
		}
		else if (emotion == 3){
			iterator = 59;
		}
		else if (emotion == 4){
			iterator = 25;
		}
		else if (emotion == 5){
			iterator = 69;
		}
		else if (emotion == 6){
			iterator = 28;
		}
		else if (emotion == 7){
			iterator = 83;
		}

		int counter = 0;
		int wrong = 0;
		int error = 0;
		for (int i = 0; i < iterator; i++){
			ifstream result;
			result.open("C:\\Users\\Dell\\Downloads\\libsvm-3.20\\libsvm-3.20\\windows\\results\\result" + to_string(i) + ".txt");
			if (result.is_open()){
				string data;
				result >> data;
				int dataint = stoi(data);
				if (dataint == 1){
					counter++;
				}
				else if ((dataint == 0)||(dataint == -1)){
					wrong++;
				}
				else{
					error++;
				}
			}
		}
		
		cout << "Correct: " << to_string(counter) << " | wrong: " << to_string(wrong) << " | error: " << to_string(error) << endl;
		sleepcp(5000);

	}

	return 0;
}