sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
echo -e 'Section "ServerFlags"\n\tOption "MaxClients" "2048"\nEndSection\n' \
    | sudo tee /etc/X11/xorg.conf.d/99-maxclients.conf