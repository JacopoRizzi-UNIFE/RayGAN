### Info su come convertire correttamente i dataset
***
### *Oxford Radar RobotCar Dataset*  
**Format:** [1,0,-2,i]  
**Vertical:** Linear  
**Fields:**  `[0][i]`  
**File:** `bin_file = bin_file.reshape((num_fields, -1))`  
Per convertire Oxford Radar RobotCar Dataset bisogna invertire colonne e righe.
***
### *Da Unity*
**Format:** [0,2,1]  
**Vertical:** Linear  
**Fields:**  `[i][0]`  
**File:** `bin_file = bin_file.reshape((-1, num_fields))`  
Rimpicciolire un po' il `self.h_fov` e il `self.v_l_fov`, tipo:  
`# Finita la calibrazione: `  
`self.v_fov_l -= 0.01`  
`self.h_fov -= 0.01`
***
### *SeeingThrougFog*
**Format:** [1,0,2,i,r]  
**Vertical:** Distribution  
**Fields:**  `[i][0]`  
**File:** `bin_file = bin_file.reshape((-1, num_fields))`
***
### *RADIATE*
**Format:** [0,1,2,i,r]  
**Vertical:** Distribution  
**Fields:**  `[i][0]`  
**File:** `bin_file = bin_file.reshape((-1, num_fields))`
***
### *NuScenes*
**Format:** [0,1,2,i,r]  
**Vertical:** Linear  
**Fields:**  `[i][0]`  
**File:** `bin_file = bin_file.reshape((-1, num_fields))`
***
### *Cadcd*
**Format:** [0,1,2,i] OPPURE [0,1,2]  
**Vertical:** Distribution  ==> `selected_row = 14`  
**Fields:**  `[i][0]`  
**File:** `bin_file = bin_file.reshape((-1, num_fields))`  
Rimpicciolire un po' il `self.h_fov` e il `self.v_l_fov`, tipo:  
`# Finita la calibrazione: `  
`self.v_fov_l -= 0.01`  
`self.h_fov -= 0.01`
### *WADS*
**Format:** [1,0,2,i]  
**Vertical:** Distribution  ==> `selected_row = 5`  
**Fields:**  `[i][0]`  
**File:** `bin_file = bin_file.reshape((-1, num_fields))`  
Rimpicciolire un po' il `self.h_fov` e il `self.v_l_fov`, tipo:  
`# Finita la calibrazione: `  
`self.v_fov_l -= 0.01`  
`self.h_fov -= 0.01`
### *KITTI*
**Format:** [0,1,2,i]  
**Vertical:** Linear  
**Fields:**  `[i][0]`  
**File:** `bin_file = bin_file.reshape((-1, num_fields))`  
Rimpicciolire un po' il `self.h_fov` e il `self.v_l_fov`, tipo:  
**HARDCODDA il H-FOV dopo la calibrazione:**     
`self.h_fov = math.pi * 2.0`