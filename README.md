Read this to setup lyricsgenius:

https://lyricsgenius.readthedocs.io/en/master/setup.html

To use, create a user_token.yaml file formatted as such:

token: YOURTOKEN


---

Development log:

4/22/2024
I had to follow instructions here: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell#enable-long-paths-in-windows-10-version-1607-and-later

Specifically running the following in Poweshell with Administrator rights:

New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

Because while pip installing transformers, datasets packages I got: 

ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'C:\\Users\\Leen\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages\\transformers\\models\\deprecated\\trajectory_transformer\\convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch.py'
HINT: This error might have occurred since this system does not have Windows Long Path support enabled. You can find information on how to enable this at https://pip.pypa.io/warnings/enable-long-paths

I also misunderstood what the nltk Vader sentiment analysis actually does-- Vader rates observations based on polarity, whereas I'm looking for emotion detection. 


