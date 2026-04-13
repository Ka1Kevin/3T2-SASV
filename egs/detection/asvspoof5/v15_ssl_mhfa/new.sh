# 想法：找到我的文件，把ref用“，”隔开，把test用“#”隔开
awk 'NR>1 {print $1"#"$4, $5}' /export/fs05/ktan17/new_file/ASVspoof5/train/1v1.tsv \
    > /export/fs05/ktan17/ssl/spk2lab

awk -F ' +' '$9=="bonafide"{print $1, $2}' /export/fs05/arts/dataset/ASVspoof5/ASVspoof5.train.tsv \
  | sort -k1,1 \
  | awk '{
      spk=$1; utt=$2;
      if (spk in arr) {
          arr[spk]=arr[spk]","utt;
      } else {
          arr[spk]=utt;
      }
  }
  END {
      for (spk in arr) {
          print spk, arr[spk];
      }
  }' > /export/fs05/ktan17/ssl/spk2utt

./tools/utt2lab_to_lab2utt.pl /export/fs05/ktan17/ssl/spk2lab \
	>/export/fs05/ktan17/ssl/lab2spk