ffmpeg -hide_banner -y \
  -fflags +genpts \
  -avoid_negative_ts make_zero \
  -i "../dut-1-hq-creamy_detail_10-420-50p@22Mbps-hevc.ts" \
  -c copy \
  -map 0 \
  -muxdelay 0 -muxpreload 0 \
  -f hls \
  -hls_time "6" \
  -hls_playlist_type vod \
  -hls_segment_type mpegts \
  -hls_segment_filename "dut-1-hq-creamy_detail_10-420-50p@22Mbps-hevc_%03d.ts" \
  -hls_flags independent_segments \
  "dut-1-hq-creamy_detail_10-420-50p@22Mbps-hevc"