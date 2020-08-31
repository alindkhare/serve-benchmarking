package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"time"
)

func unixMilli(t time.Time) float64 {
	return float64(t.Round(time.Millisecond).UnixNano() / (int64(time.Millisecond) / int64(time.Nanosecond)))
}

func MakeRequest(client *http.Client, url string, values map[string]string, ch chan<- string) {
	// values := map[string]byte{"data": username}

	start := time.Now()
	// current_time := unixMilli(start)
	// current_time += deadline_ms
	// current_time_str := strconv.FormatFloat(current_time, 'E', -1, 64)
	// values["absolute_slo_ms"] = current_time_str
	jsonValue, _ := json.Marshal(values)
	resp, _ := client.Post(url, "application/json", bytes.NewBuffer(jsonValue))
	secs := time.Since(start).Seconds()
	body, _ := ioutil.ReadAll(resp.Body)
	ch <- fmt.Sprintf("%.2f elapsed with response length: %s %s", secs, body, url)
}
func main() {
	runtime.GOMAXPROCS(5)
	ch := make(chan string)
	image_file_path := os.Args[1]
	endpoint := os.Args[2]
	arrival_curve := os.Args[3:]
	imgFile, err := os.Open(image_file_path)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	defer imgFile.Close()

	tr := &http.Transport{
		MaxIdleConnsPerHost: 30,
		MaxConnsPerHost:     31,
	}
	client := &http.Client{Transport: tr}

	fInfo, _ := imgFile.Stat()
	var size int64 = fInfo.Size()
	buf := make([]byte, size)
	fReader := bufio.NewReader(imgFile)
	fReader.Read(buf)
	imgBase64Str := base64.StdEncoding.EncodeToString(buf)

	time.Sleep(10 * time.Millisecond)
	fmt.Println("Start firing to the client")
	fmt.Println(len(arrival_curve))
	start := time.Now()
	for i := 0; i < len(arrival_curve); i++ {
		// time.Sleep(12195 * time.Microsecond)
		values := map[string]string{"data": imgBase64Str}
		time_ms, err_time := strconv.ParseFloat(arrival_curve[i], 64)
		if err_time != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		time.Sleep(time.Duration(time_ms) * time.Millisecond)
		//fmt.Println("done sleep")
		//time.Sleep(time.Duration(time_ms) * time.Microsecond)
		// values := map[string]string{"data": imgBase64Str}
		go MakeRequest(client, "http://127.0.0.1:8000"+endpoint, values, ch)
	}
	fmt.Println("Fired all the queries now waiting for them")
	for i := 0; i < len(arrival_curve); i++ {
		fmt.Println(<-ch)
		s := fmt.Sprintf("query complete %d", i)
		fmt.Println(s)
	}
	fmt.Printf("%.2fs elapsed\n", time.Since(start).Seconds())
}
