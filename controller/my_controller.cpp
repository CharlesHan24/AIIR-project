
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <../c/webots/camera.h>
#include <../c/webots/motor.h>
#include <../c/webots/robot.h>
#include <../c/webots/utils/system.h>
#include <../c/webots/distance_sensor.h>
#include <my_detector.h>

#define TIME_STEP 64
#define MAX_SPEED 10
#define MAX_STATUS 1000000
#define DIS_PER_TIME 1
#define TURN_TIME 8

int dir[MAX_STATUS], dis[MAX_STATUS], choice[MAX_STATUS];

WbDeviceTag wheels[4];
char wheels_names[4][8] = {"wheel1", "wheel2", "wheel3", "wheel4"};
WbDeviceTag ps[4];

void turn_left() {
	double left_speed  =  -0.1*MAX_SPEED;
	double right_speed =  0.1*MAX_SPEED;
	wb_motor_set_velocity(wheels[0], left_speed);
	wb_motor_set_velocity(wheels[1], left_speed);
	wb_motor_set_velocity(wheels[2], right_speed);
	wb_motor_set_velocity(wheels[3], right_speed);
}
void turn_right() {
	
	double left_speed  =  0.1*MAX_SPEED;
	double right_speed =  -0.1*MAX_SPEED;
	wb_motor_set_velocity(wheels[0], left_speed);
	wb_motor_set_velocity(wheels[1], left_speed);
	wb_motor_set_velocity(wheels[2], right_speed);
	wb_motor_set_velocity(wheels[3], right_speed);
}

void go_forward(){
	wb_motor_set_velocity(wheels[0], 0.5*MAX_SPEED);
	wb_motor_set_velocity(wheels[1], 0.5*MAX_SPEED);
	wb_motor_set_velocity(wheels[2], 0.5*MAX_SPEED);
	wb_motor_set_velocity(wheels[3], 0.5*MAX_SPEED);
}

int main(){
          Interface interf = Interface("template5.png");
	wb_robot_init();
	//printf("gggg\n");

	//打开摄像头
	WbDeviceTag camera[3];
	const char *filenames[3] = {"01.png", "02.png", "03.png"};
	char camera_names[3][8] = {"camera1", "camera2", "camera3"};
	for (int i = 0; i < 3 ; i++){
		camera[i] = wb_robot_get_device(camera_names[i]);
		wb_camera_enable(camera[i], TIME_STEP);
	}
	//打开摄像头完成
	
	
	for (int i = 0; i < 4 ; i++){
		wheels[i] = wb_robot_get_device(wheels_names[i]);
		wb_motor_set_position(wheels[i], INFINITY);
		wb_motor_set_velocity(wheels[i], 0);
	}
	
	ps[0] = wb_robot_get_device("ds_left");
	wb_distance_sensor_enable(ps[0], TIME_STEP);
	ps[1] = wb_robot_get_device("ds_right");
	wb_distance_sensor_enable(ps[1], TIME_STEP);
	ps[2] = wb_robot_get_device("ds_front_left");
	wb_distance_sensor_enable(ps[2], TIME_STEP);
	ps[3] = wb_robot_get_device("ds_front_right");
	wb_distance_sensor_enable(ps[3], TIME_STEP);
	
	int pause_counter = 40;
	int j = 0, k = 0, i;
	bool r = false;
	double obstacle_1;
	
	while (wb_robot_step(TIME_STEP) != -1) {
		double ps_values[4];
		for (int i = 0; i < 4 ; i++){
			ps_values[i] = wb_distance_sensor_get_value(ps[i]);
			//printf("%lf ",ps_values[i]);
		}
 
		bool left_obstacle = (ps_values[0] < 150.0);
		bool right_obstacle = (ps_values[1] < 150.0);
		if(k) right_obstacle = (ps_values[1] < 250.0);
		bool front_obstacle = (ps_values[2] < 150.0 || ps_values[3] < 150.0); 

		if(j){
			go_forward();
			--j;
		}

		else if(r && ps_values[1] > obstacle_1){
			go_forward();
			j = 3;
			r = false;
		}
		else if (!right_obstacle) {
                double left_speed  =  0.05*MAX_SPEED;
                double right_speed =  -0.05*MAX_SPEED;
                turn_right();
        	    k = 0;
        	    r = true;
        	    obstacle_1 = ps_values[1];
		}
		else if(!front_obstacle) {
                  go_forward();
                  r = false;
        	}
		else {
    		     turn_left();
                	k = 1;
                	r = false;
              }
           //printf("\n");
		 
            if(pause_counter == 0)
           {
           char *filepath;
           //left_speed = 0;
           //right_speed = 0;
           for(int j = 0; j < 3; ++j){
           
             #ifdef _WIN32
                const char *user_directory = wbu_system_short_path(wbu_system_getenv("USERPROFILE"));
                filepath = (char *)malloc(strlen(user_directory) + 16);
                strcpy(filepath, user_directory);
                strcat(filepath, "\\");   
             #else
               //const char* user_directory = "home/hanwenchen/controllers/my_controller2/";
               //filepath = (char*)malloc(strlen(user_directory) + 20);
               //strcpy(filepath, user_directory);
               //printf("%s\n", filepath);
                const char *user_directory = wbu_system_getenv("HOME");
                filepath = (char *)malloc(strlen(user_directory) + 16);
                strcpy(filepath, user_directory);
                strcat(filepath, "/");  
             #endif
           
             strcat(filepath, filenames[j]);
             //printf("%s\n", filepath);
             wb_camera_save_image(camera[j], filepath, 100);
             free(filepath);
            }
            pause_counter = 20;
            double res[3];
            res[0] = 0;
            //res[0] = interf.query_TF("../../01.png");
            res[1] = interf.query_TF("../../02.png");
            if ((res[0] < 0.6) && (res[1] < 0.6));
              //printf("%f %f\n", res[0], res[1]);
            else{
              double res2[3];
              wb_robot_step(TIME_STEP);
              wb_robot_step(TIME_STEP);
              res2[0] = interf.query_TF("../../01.png");
              res2[1] = interf.query_TF("../../02.png");
              if ((res[0] > 2) && (res2[0] > 0.6)){
                printf("ok, I find it!\n");
                printf("In front of me\n");
                wb_robot_step(TIME_STEP);
                wb_robot_cleanup();
                while(1);
                return 0;
              }
              else if ((res[1] > 0.6) && (res2[1] > 0.6)){
                printf("ok, I find it!\n");
                printf("On the left side\n");
                wb_robot_step(TIME_STEP);
                wb_robot_cleanup();
                while(1);
                return 0;
              }
              //else
                //printf("%f %f\n", res[0], res[1]);
             }
           }
           else --pause_counter;
	}
	wb_robot_cleanup();
	return 0;  
}
