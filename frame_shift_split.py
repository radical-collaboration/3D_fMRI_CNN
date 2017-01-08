import os
import numpy as np
import nibabel as nb
import glob
import scipy.io as sio

path = '/Users/JumanaDakka/3D_CNN_fMRI/input_data_nii'


def sites_labels():
    mat = sio.loadmat('SubjectsID_final.mat')

    site_ID_0009 = mat['SubjectsID'][0, 0]['SCZ_site0009_']['ID']
    site_labels_0009 = mat['SubjectsID'][0, 0]['SCZ_site0009_']['labels']

    site_ID_0018 = mat['SubjectsID'][0, 0]['SCZ_site0018_']['ID']
    site_labels_0018 = mat['SubjectsID'][0, 0]['SCZ_site0018_']['labels']

    site_ID_0006 = mat['SubjectsID'][0, 0]['SCZ_site0006_']['ID']
    site_labels_0006 = mat['SubjectsID'][0, 0]['SCZ_site0006_']['labels']

    site_ID_0003 = mat['SubjectsID'][0, 0]['SCZ_site0003_']['ID']
    site_labels_0003 = mat['SubjectsID'][0, 0]['SCZ_site0003_']['labels']

    site_ID_0010 = mat['SubjectsID'][0, 0]['SCZ_site0010_']['ID']
    site_labels_0010 = mat['SubjectsID'][0, 0]['SCZ_site0010_']['labels']

    site_ID_0009 = site_ID_0009[0, 0]
    site_ID_0009 = site_ID_0009.tolist()

    site_ID_0018 = site_ID_0018[0, 0]
    site_ID_0018 = site_ID_0018.tolist()

    site_ID_0006 = site_ID_0006[0, 0]
    site_ID_0006 = site_ID_0006.tolist()

    site_ID_0003 = site_ID_0003[0, 0]
    site_ID_0003 = site_ID_0003.tolist()

    site_ID_0010 = site_ID_0010[0, 0]
    site_ID_0010 = site_ID_0010.tolist()

    list_subject_ID = np.append(site_ID_0009, site_ID_0018)
    list_subject_ID = np.append(list_subject_ID, site_ID_0006)
    list_subject_ID = np.append(list_subject_ID, site_ID_0003)
    list_subject_ID = np.append(list_subject_ID, site_ID_0010)

    site_labels_0009 = site_labels_0009[0, 0]
    site_labels_0009 = site_labels_0009.tolist()

    site_labels_0018 = site_labels_0018[0, 0]
    site_labels_0018 = site_labels_0018.tolist()

    site_labels_0006 = site_labels_0006[0, 0]
    site_labels_0006 = site_labels_0006.tolist()

    site_labels_0003 = site_labels_0003[0, 0]
    site_labels_0003 = site_labels_0003.tolist()

    site_labels_0010 = site_labels_0010[0, 0]
    site_labels_0010 = site_labels_0010.tolist()

    list_labels = np.append(site_labels_0009, site_labels_0018)
    list_labels = np.append(list_labels, site_labels_0006)
    list_labels = np.append(list_labels, site_labels_0003)
    list_labels = np.append(list_labels, site_labels_0010)

    return list_subject_ID, list_labels


# print list_subject_ID
# print list_labels

def parse_filename(filename):
    subject_ID = os.path.basename(filename)
    subject_ID = os.path.splitext(subject_ID)
    subject_ID = subject_ID[0]
    subject_ID = os.path.splitext(subject_ID)
    subject_ID = subject_ID[0]
    subject_ID = subject_ID[:12]
    # print subject_ID
    return subject_ID


def parse_filename_storing(filename):
    subject_ID = os.path.basename(filename)
    subject_ID = os.path.splitext(subject_ID)
    subject_ID = subject_ID[0]
    subject_ID = os.path.splitext(subject_ID)
    subject_ID_string = subject_ID[0]
    return subject_ID_string


def return_label(list_subject_ID, list_labels, subject_ID):
    for index, item in enumerate(list_subject_ID):
        if item == subject_ID:
            return list_labels[index]


def load_data_split_frames(list_subject_ID, list_labels):
    for filename in glob.glob(os.path.join(path, '*.nii.gz')):
        img = nb.load(filename)
        subject_ID = parse_filename(filename)
        label = return_label(list_subject_ID, list_labels, subject_ID)
        total_frames = img.shape[(3)]
        interval = 10
        i = 0
        frame_count = 0
        if label == -1:
            while i < total_frames:
                frames = img.dataobj[..., i:i + interval]
                i = i + interval / 2
                # print frame_count

                '''storing the collection of frames in format: label(1=normal, -1 schizophrenic)_framecount(0-27)_subjectID'''

                subject_ID_string = parse_filename_storing(filename)
                frames = nb.Nifti1Image(frames, affine=np.eye(4))
                subject_ID_string = parse_filename_storing(filename)
                string_name = str(label) + '_' + str(frame_count) + '_' + subject_ID_string
                nb.save(frames, os.path.join('normal', string_name))
                frame_count = frame_count + 1

        if label == 1:
            while i < total_frames:
                frames = img.dataobj[..., i:i + interval]
                i = i + interval / 2
                # print frame_count

                '''storing the collection of frames in format: label(1=normal, -1 schizophrenic)_framecount(0-27)_subjectID'''

                frames = nb.Nifti1Image(frames, affine=np.eye(4))
                subject_ID_string = parse_filename_storing(filename)
                string_name = str(label) + '_' + str(frame_count) + '_' + subject_ID_string
                nb.save(frames, os.path.join('schizophrenic', string_name))
                frame_count = frame_count + 1


def main():
    [list_subject_ID, list_labels] = sites_labels()
    load_data_split_frames(list_subject_ID, list_labels)


def test():
    for filename in glob.glob(os.path.join(path, '*.nii.gz')):
        img = nb.load(filename)
        subject_ID_string = parse_filename_storing(filename)
        total_frames = img.shape[(3)]
        interval = 10
        i = 0
        frame_count = 0
        while i < total_frames:
            frames = img.dataobj[..., i:i + interval]
            i = i + interval / 2
            print frame_count
            frames = nb.Nifti1Image(frames, affine=np.eye(4))
            string_name = str(frame_count) + '_' + subject_ID_string
            nb.save(frames, os.path.join('build', string_name))
            frame_count = frame_count + 1


main()
# sites_labels2()

# test()


'''


def sites_labels2():

	mat=sio.loadmat('SubjectsID_final.mat')
	list_subject_ID = None
	list_labels = None
	for key in mat['SubjectsID'][0,0]:
		site_ID = mat['SubjectsID'][0,0][key]['ID'][0,0].tolist()
		site_labels = mat['SubjectsID'][0,0][key]['labels'][0,0].tolist()
	print site_ID
	print site_labels	

vol1= img.dataobj[..., 0:10]

proxy_img=img
vol1= proxy_img.dataobj[..., 0:10]
vol2=proxy_img.dataobj[..., 5:15]
print vol1.shape 
print vol2.shape


print(img)
print img.shape

affine = img.affine
print affine 	


'''
