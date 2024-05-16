#include <unordered_map>
#include <string>

class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        /*
        Sliding window of occurrences

        s2 contains a permutation of s1 if:
            s1.size() <= s2.size()
            there exists a part in s2 such that the occurrence window (length equal to s1.size()) of s2 matches that of s1

        3 O(n) passes
        Present in s1 -> tally once
        Present in s2 (right) -> untally once. if value is now 0, then erase?
        Shifting window left -> if character is present in map, re-tally once
        If at the end, all tallies are 0, return true
        Otherwise, return false
        */
        if(s1.size() > s2.size()){
            return false;
        }

        unordered_map<char, int> m;
        int n = s1.size();
        for(int i = 0; i < n; i++){
            if(m.find(s1[i]) != m.end()){
                m[s1[i]]++;
            }else{
                m[s1[i]] = 1;
            }
        }
        
        // for (const auto& pair : m) {
        //     cout << pair.first << ":" << pair.second << endl;
        // }
        int left = 0, right = 0;        
        for(right = 0; right < s2.size(); right++){
            // cout << m.size() << endl;
            if(m.find(s2[right]) != m.end()){
                m[s2[right]]--;
                // cout << s2[right] << endl;
            }

            for(const auto& pair: m){
                // cout << pair.first << ":" << pair.second << endl;
                if(pair.second != 0){
                    // cout << pair.first << "-" << pair.second << endl;
                    break;
                }
                return true;
            }

            // cout << left << " " << right << endl;
            if((1 + right - left) == n){ // if the window is equal to s1's size
                // cout << s2[left] << endl;
                if(m.find(s2[left]) != m.end()){ // if the leftmost character was tallied AND we are not satisfied
                    m[s2[left]]++;    
                }
                left++;
            }
        }

        // return if all key-value pairs in m are 0
        // for(const auto& pair: m){
        //     auto it = m.find(pair.first);
        //     if(pair.second != 0){
        //         return false;
        //     }
        // }
        // return true;
        return false;
    }
};