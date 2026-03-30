ffmpeg -f rawvideo -pix_fmt yuv444p -s:v 1920x1080 -i creamy_detail_1X-50p-444p-1920x1080.yuv -frames:v 1 1X_first_frame.png &&\
ffmpeg -f rawvideo -pix_fmt yuv444p -s:v 1920x1080 -i creamy_detail_1Y-60p-444p-1920x1080.yuv -frames:v 1 1Y_first_frame.png &&\
ffmpeg -f rawvideo -pix_fmt yuv444p -s:v 1920x1080 -i creamy_detail_2X-50p-444p-1920x1080.yuv -frames:v 1 2X_first_frame.png &&\
ffmpeg -f rawvideo -pix_fmt yuv444p -s:v 1920x1080 -i creamy_detail_2Y-60p-444p-1920x1080.yuv -frames:v 1 2Y_first_frame.png 