# Leaderboard

## Automatic Speech Recognition (ASR)


<table>

  <thead>
    <tr>
      <th>Datasets</th>
      <th>Model</th>
      <th>Performance (WER↓)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5"><b>LibriSpeech</b><br>test-clean | test-other</td>
      <td>Qwen2-Audio-base</td>
      <td>1.74 | 4.04</td>
    </tr>
    <tr>
      <td>Baichuan-base</td>
      <td>3.02 | 6.04</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>3.19 | 10.67</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>2.37 | 4.21</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>1.28 | 2.42</b></td>
    </tr>
    <tr>
      <td rowspan="5"><b>Fleurs</b><br>zh | en</td>
      <td>Qwen2-Audio-base</td>
      <td>3.63 | 5.20</td>
    </tr>
    <tr>
      <td>Baichuan-base</td>
      <td>4.15 | 8.07</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>4.26 | 8.56</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>2.92 | <b>4.17</b></td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>2.69</b> | 4.44</td>
    </tr>
    <tr>
      <td rowspan="5"><b>AISHELL-1</b></td>
      <td>Qwen2-Audio-base</td>
      <td>1.52</td>
    </tr>
    <tr>
      <td>Baichuan-base</td>
      <td>1.93</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>2.14</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>1.13</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>0.60</b></td>
    </tr>
    <tr>
      <td rowspan="5"><b>AISHELL-2</b> ios</td>
      <td>Qwen2-Audio-base</td>
      <td>3.08</td>
    </tr>
    <tr>
      <td>Baichuan-base</td>
      <td>3.87</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>3.89</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td><b>2.56</b></td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>2.56</b></td>
    </tr>
    <tr>
      <td rowspan="5"><b>WenetSpeech</b><br>test-meeting | test-net</td>
      <td>Qwen2-Audio-base</td>
      <td>8.40 | 7.64</td>
    </tr>
    <tr>
      <td>Baichuan-base</td>
      <td>13.28 | 10.13</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>10.83 | 9.47</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>7.71 | 6.04</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>6.28 | 5.37</b></td>
    </tr>
    <tr>
      <td rowspan="5"><b>Kimi-ASR Internal Testset</b><br>subset1 | subset2</td>
      <td>Qwen2-Audio-base</td>
      <td>2.31 | 3.24</td>
    </tr>
    <tr>
      <td>Baichuan-base</td>
      <td>3.41 | 5.60</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>2.82 | 4.74</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>1.53 | 2.68</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>1.42 | 2.44</b></td>
    </tr>
  </tbody>
</table>

## Audio Understanding

<table>
  <thead>
    <tr>
      <th>Datasets</th>
      <th>Model</th>
      <th>Performance↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><b>MMAU</b><br>music | sound | speech</td>
      <td>Qwen2-Audio-base</td>
      <td>58.98 | 69.07 | 52.55</td>
    </tr>
    <tr>
      <td>Baichuan-chat</td>
      <td>49.10 | 59.46 | 42.47</td>
    </tr>
    <tr>
      <td>GLM-4-Voice</td>
      <td>38.92 | 43.54 | 32.43</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>49.40 | 53.75 | 47.75</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td><b>62.16</b> | 67.57 | 53.92</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td>61.68 | <b>73.27</b> | <b>60.66</b></td>
    </tr>
    <tr>
      <td rowspan="5"><b>ClothoAQA</b><br>test | dev</td>
      <td>Qwen2-Audio-base</td>
      <td>71.73 | 72.63</td>
    </tr>
    <tr>
      <td>Baichuan-chat</td>
      <td>48.02 | 48.16</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>45.84 | 44.98</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td><b>72.86</b> | 73.12</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td>71.24 | <b>73.18</b></td>
    </tr>
    <tr>
      <td rowspan="5"><b>VocalSound</b></td>
      <td>Qwen2-Audio-base</td>
      <td>93.82</td>
    </tr>
    <tr>
      <td>Baichuan-base</td>
      <td>58.17</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>28.58</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>93.73</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>94.85</b></td>
    </tr>
    <tr>
      <td rowspan="5"><b>Nonspeech7k</b></td>
      <td>Qwen2-Audio-base</td>
      <td>87.17</td>
    </tr>
    <tr>
      <td>Baichuan-chat</td>
      <td>59.03</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>21.38</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>69.89</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>93.93</b></td>
    </tr>
    <tr>
      <td rowspan="5"><b>MELD</b></td>
      <td>Qwen2-Audio-base</td>
      <td>51.23</td>
    </tr>
    <tr>
      <td>Baichuan-chat</td>
      <td>23.59</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>33.54</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>49.83</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>59.13</b></td>
    </tr>
    <tr>
      <td rowspan="5"><b>TUT2017</b></td>
      <td>Qwen2-Audio-base</td>
      <td>33.83</td>
    </tr>
    <tr>
      <td>Baichuan-base</td>
      <td>27.9</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>7.41</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>43.27</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>65.25</b></td>
    </tr>
    <tr>
      <td rowspan="5"><b>CochlScene</b><br>test | dev</td>
      <td>Qwen2-Audio-base</td>
      <td>52.69 | 50.96</td>
    </tr>
    <tr>
      <td>Baichuan-base</td>
      <td>34.93 | 34.56</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>10.06 | 10.42</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>63.82 | 63.82</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>79.84 | 80.99</b></td>
    </tr>
  </tbody>
</table>

## Audio-to-Text Chat

<table>
  <thead>
    <tr>
      <th>Datasets</th>
      <th>Model</th>
      <th>Performance↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><b>OpenAudioBench</b><br>AlpacaEval | Llama Questions |<br>Reasoning QA | TriviaQA | Web Questions</td>
      <td>Qwen2-Audio-chat</td>
      <td>57.19 | 69.67 | 42.77 | 40.30 | 45.20</td>
    </tr>
    <tr>
      <td>Baichuan-chat</td>
      <td>59.65 | 74.33 | 46.73 | 55.40 | 58.70</td>
    </tr>
    <tr>
      <td>GLM-4-Voice</td>
      <td>57.89 | 76.00 | 47.43 | 51.80 | 55.40</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>56.53 | 72.33 | 60.00 | 56.80 | <b>73.00</b></td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>72.76 | 75.33 | <b>63.76</b> | 57.06 | 62.80</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>75.73</b> | <b>79.33</b> | 58.02 | <b>62.10 </b> | 70.20</td>
    </tr>
    <tr>
      <td rowspan="6"><b>VoiceBench</b><br>AlpacaEval | CommonEval |<br>SD-QA | MMSU</td>
      <td>Qwen2-Audio-chat</td>
      <td>3.69 | 3.40 | 35.35 | 35.43</td>
    </tr>
    <tr>
      <td>Baichuan-chat</td>
      <td>4.00 | 3.39 | 49.64 | 48.80</td>
    </tr>
    <tr>
      <td>GLM-4-Voice</td>
      <td>4.06 | 3.48 | 43.31 | 40.11</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>3.99 | 2.99 | 46.84 | 28.72</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>4.33 | 3.84 | 57.41 | 56.38</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>4.46</b> | <b>3.97</b> | <b>63.12</b> | <b>62.17</b></td>
    </tr>
    <tr>
      <td rowspan="6"><b>VoiceBench</b><br>OpenBookQA | IFEval |<br>AdvBench | Avg</td>
      <td>Qwen2-Audio-chat</td>
      <td>49.01 | 22.57 | 98.85 | 54.72</td>
    </tr>
    <tr>
      <td>Baichuan-chat</td>
      <td>63.30 | 41.32 | 86.73 | 62.51</td>
    </tr>
    <tr>
      <td>GLM-4-Voice</td>
      <td>52.97 | 24.91 | 88.08 | 57.17</td>
    </tr>
    <tr>
      <td>StepAudio-chat</td>
      <td>31.87 | 29.19 | 65.77 | 48.86</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td>
      <td>79.12 | 53.88 | 99.62 | 72.83</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
      <td><b>83.52</b> | <b>61.10</b> | <b>100.00</b> | <b>76.93</b></td>
    </tr>
  </tbody>
</table>



## Updates
- 2025-04-25: Initial leaderboard created

