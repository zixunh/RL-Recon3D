sudo nvidia-xconfig -a --virtual=1920x1080
echo -e 'Section "ServerFlags"\n\tOption "MaxClients" "2048"\nEndSection\n' \
    | sudo tee /usr/share/X11/xorg.conf.d/99-maxclients.conf