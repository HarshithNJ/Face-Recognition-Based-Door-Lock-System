int t=12;

void setup()
{
  pinMode(t,OUTPUT);
}

void loop()
{
  while(1)
  {
    digitalWrite(12,HIGH);
    delay(500);
    digitalWrite(12,LOW);
    delay(500);
  }
}