
import os

print("Creating dataset text files")
#Create text files for train and val 
for ds in [1,2,3,4,5,6,7,11,12]:
  #image_dir = "/projects/mines/working_mount/processed_sonar/new_data/" + "{}/range5000/k-8".format(ds)
  gt_dir = "/Users/Josh/mines_ground_truth/sonar/range5000/{}".format(ds)
  full_gt_names = os.listdir(gt_dir)
  dp_names = []
  for name in full_gt_names:
    print("name: ", name)
    split = name.split(".")
    dp_name = split[0] + "." + split[1] + "." + split[2]
    dp_names.append(dp_name)
  print("writing file for dataset {}".format(ds))
  with open("/Users/Josh/mines_file_names/sonar_ds_{}_list.txt".format(ds), "w") as f:
      for name in dp_names:
          f.write("{}\n".format(name))
