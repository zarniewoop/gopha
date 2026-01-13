ffprobe -hide_banner -loglevel error \
  -select_streams v:0 \
  -skip_frame nokey \
  -show_frames \
  -show_entries frame=best_effort_timestamp_time \
  -of csv=p=0 dut-1-hq-tennis_1080p50@22Mbps-hevc.ts \
| awk '
  NR==1 {prev=$1; print "t=" $1 "  (first keyframe)"; next}
  {
    dt=$1-prev;
    print "t=" $1 "  dt=" dt;
    sum+=dt; n+=1; prev=$1
  }
  END {
    if (n>0) print "avg_dt=" (sum/n) " seconds over " n " intervals"
  }'
