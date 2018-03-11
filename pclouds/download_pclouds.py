import os
import zipfile
import urllib.request

source_url = 'http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip'
target_dir = os.path.dirname(os.path.abspath(__file__))
target_file = os.path.join(target_dir, 'pclouds.zip')

downloaded = 0
def show_progress(count, block_size, total_size):
    global downloaded
    downloaded += block_size
    print('downloading ... %d%%' % round(((downloaded*100.0) / total_size)), end='\r')

print('downloading ... ', end='\r')
urllib.request.urlretrieve(source_url, filename=target_file, reporthook=show_progress)
print('downloading ... done')

print('unzipping ...', end='\r')
zip_ref = zipfile.ZipFile(target_file, 'r')
zip_ref.extractall(target_dir)
zip_ref.close()
os.remove(target_file)
print('unzipping ... done')
